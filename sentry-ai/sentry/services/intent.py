# sentry/services/intent.py
"""
Query Intent Classification Service
====================================

Regex-based intent classifier that routes queries to appropriate
data sources (Trace VectorDB vs Raw Logs VectorDB vs Helix Anomalies).

Based on research module with improvements:
- Comprehensive regex patterns with edge cases
- Weighted confidence scoring
- VectorDB routing hints for optimal query execution
- Query subject extraction for targeted search

Architecture:
    User Query → IntentClassifier.classify() → IntentResult
                                                  ↓
                              ┌──────────────────┴──────────────────┐
                              ↓                                      ↓
                    Trace VectorDB                           Raw Logs DB
                    (frequencies)                            (context)
"""

import re
import logging
from enum import Enum
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class QueryIntent(Enum):
    """
    Types of queries the system can handle.
    
    Each intent maps to a specific data source or search strategy.
    """
    FREQUENCY = "frequency"      # How often does X happen? → Transition stats
    WHY = "why"                  # Why did X happen? → Raw logs with context
    ERROR = "error"              # What errors occurred? → Helix anomalies
    ANOMALY = "anomaly"          # Anomaly detection queries → Helix
    SIMILAR = "similar"          # Find similar patterns → Similarity search
    TIMELINE = "timeline"        # When did X happen? → Time-ordered search
    GENERAL = "general"          # General question → Standard RAG


@dataclass(frozen=True)
class IntentResult:
    """
    Result of intent classification.
    
    Immutable dataclass for thread-safety and hashability.
    """
    intent: QueryIntent
    confidence: float           # 0.0 to 1.0
    subject: Optional[str]      # Extracted query subject
    routing_hint: str           # Which data source to prioritize
    matched_patterns: int       # Number of patterns matched


class IntentClassifier:
    """
    Improved regex-based intent classifier.
    
    Routes queries to optimal data sources based on pattern matching.
    Compiles regex patterns once at init for performance.
    
    Usage:
        classifier = IntentClassifier()
        result = classifier.classify("Why did the database timeout?")
        print(result.intent)  # QueryIntent.WHY
        print(result.routing_hint)  # "raw_logs_with_context"
    """
    
    # Pattern definitions (will be compiled at init)
    _PATTERN_DEFS: Dict[QueryIntent, List[str]] = {
        QueryIntent.FREQUENCY: [
            r'\bhow (often|many|frequently)\b',
            r'\bfrequency\b',
            r'\bcount\s*(of)?\b',
            r'\bhow many times\b',
            r'\boccurrences?\b',
            r'\bstatistics?\b',
            r'\brate\s*(of)?\b',
            r'\bprobabilit(y|ies)\b',
            r'\btransition(s)?\b',
            r'\bpercentage\b',
            r'\bratio\b',
            r'\baverage\b',
            r'\btrend(s)?\b',
            r'\bmetric(s)?\b',
        ],
        QueryIntent.WHY: [
            r'\bwhy\b',
            r'\bwhat caused\b',
            r'\bexplain\b',
            r'\breason\s*(for|why)?\b',
            r'\broot cause\b',
            r'\bwhat happened\b',
            r'\bwhat led to\b',
            r'\bhow did .* happen\b',
            r'\bcontext\s*(of|for)?\b',
            r'\bexplanation\b',
            r'\bunderstand(ing)?\b',
            r'\bdiagnos(e|is|tic)\b',
            r'\binvestigat(e|ion)\b',
            r'\banalyz(e|is)\b',
        ],
        QueryIntent.ERROR: [
            r'\berror(s)?\b',
            r'\bfailure(s)?\b',
            r'\bproblem(s)?\b',
            r'\bissue(s)?\b',
            r'\bcrash(es|ed|ing)?\b',
            r'\bexception(s)?\b',
            r'\bbug(s)?\b',
            r'\bfatal\b',
            r'\bcritical\b',
            r'\bbroken\b',
            r'\bfailed\b',
            r'\bfault(s|y)?\b',
            r'\bfailing\b',
        ],
        QueryIntent.ANOMALY: [
            r'\banomal(y|ies|ous)\b',
            r'\bunusual\b',
            r'\bsuspicious\b',
            r'\bstrange\b',
            r'\bweird\b',
            r'\boutlier(s)?\b',
            r'\bdeviation(s)?\b',
            r'\babnormal\b',
            r'\bunexpected\b',
            r'\birregular\b',
            r'\batypical\b',
        ],
        QueryIntent.SIMILAR: [
            r'\bsimilar\b',
            r'\blike this\b',
            r'\bpattern(s)?\b',
            r'\brelated\b',
            r'\bother .* like\b',
            r'\bfind .* matching\b',
            r'\bmatching\b',
            r'\bresembl(e|ing)\b',
            r'\bcompar(e|able|ison)\b',
            r'\bcorrelat(e|ed|ion)\b',
        ],
        QueryIntent.TIMELINE: [
            r'\bwhen did\b',
            r'\bwhen was\b',
            r'\btimeline\b',
            r'\bsequence\b',
            r'\border of\b',
            r'\bbefore .* (happened|occurred)\b',
            r'\bafter .* (happened|occurred)\b',
            r'\bduring\b',
            r'\btime of\b',
            r'\bhistory\b',
            r'\bchronolog(y|ical)\b',
            r'\bfirst\b.*\b(error|issue|problem)',
            r'\blast\b.*\b(error|issue|problem)',
        ],
    }
    
    # Routing hints for each intent type
    _ROUTING: Dict[QueryIntent, str] = {
        QueryIntent.FREQUENCY: "trace_vectordb",
        QueryIntent.WHY: "raw_logs_with_context",
        QueryIntent.ERROR: "helix_anomalies",
        QueryIntent.ANOMALY: "helix_anomalies",
        QueryIntent.SIMILAR: "similarity_search",
        QueryIntent.TIMELINE: "time_ordered_search",
        QueryIntent.GENERAL: "standard_rag",
    }
    
    # Patterns to remove when extracting subject
    _SUBJECT_REMOVE_PATTERNS: List[str] = [
        r'^(why|how|what|when|where|who|which)\s+',
        r'^(did|does|do|is|are|was|were|has|have|had)\s+',
        r'^(the|a|an)\s+',
        r'\?$',
        r'\bhappen(ed|s|ing)?\b',
        r'\boccur(red|s|ring)?\b',
        r'\bcause(d|s)?\b',
        r'\boften\b',
        r'\bmany\b',
        r'\btimes?\b',
        r'\bshow\s+me\b',
        r'\btell\s+me\b',
        r'\bfind\b',
        r'\blist\b',
        r'\bgive\s+me\b',
        r'\bget\b',
    ]
    
    __slots__ = ('_compiled_patterns', '_compiled_subject_patterns')
    
    def __init__(self):
        """
        Initialize classifier with compiled regex patterns.
        
        Patterns are compiled once for optimal performance during
        repeated classify() calls.
        """
        # Pre-compile all intent patterns
        self._compiled_patterns: Dict[QueryIntent, List[re.Pattern]] = {}
        for intent, patterns in self._PATTERN_DEFS.items():
            self._compiled_patterns[intent] = [
                re.compile(p, re.IGNORECASE) for p in patterns
            ]
        
        # Pre-compile subject extraction patterns
        self._compiled_subject_patterns: List[re.Pattern] = [
            re.compile(p, re.IGNORECASE) for p in self._SUBJECT_REMOVE_PATTERNS
        ]
        
        logger.debug("IntentClassifier initialized with %d intent types", 
                     len(self._compiled_patterns))
    
    def classify(self, query: str) -> IntentResult:
        """
        Classify a query and return intent with routing information.
        
        Algorithm:
        1. Normalize query (lowercase, strip)
        2. Count pattern matches for each intent
        3. Calculate confidence as (matches / total_patterns)
        4. Extract subject for targeted search
        5. Return result with routing hint
        
        Args:
            query: User's natural language query
            
        Returns:
            IntentResult with intent, confidence, subject, and routing hint
            
        Example:
            >>> classifier.classify("Why did the database timeout?")
            IntentResult(intent=QueryIntent.WHY, confidence=0.14, 
                        subject="database timeout", routing_hint="raw_logs_with_context")
        """
        if not query or not query.strip():
            return IntentResult(
                intent=QueryIntent.GENERAL,
                confidence=0.0,
                subject=None,
                routing_hint=self._ROUTING[QueryIntent.GENERAL],
                matched_patterns=0
            )
        
        query_normalized = query.lower().strip()
        
        # Score each intent by counting pattern matches
        scores: Dict[QueryIntent, int] = {}
        for intent, patterns in self._compiled_patterns.items():
            matches = sum(1 for p in patterns if p.search(query_normalized))
            scores[intent] = matches
        
        # Find best match
        total_matches = sum(scores.values())
        if total_matches == 0:
            return IntentResult(
                intent=QueryIntent.GENERAL,
                confidence=0.0,
                subject=self._extract_subject(query_normalized),
                routing_hint=self._ROUTING[QueryIntent.GENERAL],
                matched_patterns=0
            )
        
        # Get intent with highest score
        best_intent = max(scores, key=lambda k: scores[k])
        best_score = scores[best_intent]
        
        # Calculate confidence (normalized by pattern count for this intent)
        pattern_count = len(self._compiled_patterns.get(best_intent, []))
        confidence = min(best_score / max(pattern_count, 1), 1.0)
        
        result = IntentResult(
            intent=best_intent,
            confidence=confidence,
            subject=self._extract_subject(query_normalized),
            routing_hint=self._ROUTING[best_intent],
            matched_patterns=best_score
        )
        
        logger.debug("Classified query '%s' as %s (conf=%.2f, matches=%d)",
                     query[:50], best_intent.value, confidence, best_score)
        
        return result
    
    def _extract_subject(self, query: str) -> str:
        """
        Extract the main subject of the query.
        
        Removes question words, filler, and common verbs to isolate
        the core subject of investigation.
        
        Args:
            query: Normalized (lowercase) query string
            
        Returns:
            Extracted subject or original query if extraction fails
        """
        subject = query
        
        # Apply removal patterns
        for pattern in self._compiled_subject_patterns:
            subject = pattern.sub(' ', subject)
        
        # Clean up whitespace
        subject = ' '.join(subject.split()).strip()
        
        # Return original if extraction resulted in empty string
        return subject if subject else query
    
    def get_routing_hint(self, intent: QueryIntent) -> str:
        """Get the routing hint for a specific intent."""
        return self._ROUTING.get(intent, "standard_rag")
    
    def __repr__(self) -> str:
        return f"<IntentClassifier(intents={len(self._compiled_patterns)})>"


# ===== SINGLETON INSTANCE =====
_classifier: Optional[IntentClassifier] = None


def get_classifier() -> IntentClassifier:
    """
    Get the global IntentClassifier instance.
    
    Lazy-loaded singleton for efficient reuse across the application.
    """
    global _classifier
    if _classifier is None:
        _classifier = IntentClassifier()
    return _classifier
