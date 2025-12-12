# sentry/services/llm.py
"""
LLM client for talking to Ollama
Generates natural language responses based on retrieved context
"""

import logging
import time
from typing import List, Dict, Optional
import json
import ollama
from ..core.models import LogChunk
from ..core.config import settings

# Try to import tenacity for retry logic, fallback to simple retry if not installed
try:
    from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
    TENACITY_AVAILABLE = True
except ImportError:
    TENACITY_AVAILABLE = False

logger = logging.getLogger(__name__)

class LLMClient:
    """
    Client for interacting with Ollama LLM
    
    Ollama runs locally and serves models like Llama 3.
    This client formats prompts, sends requests, and parses responses.
    """
    
    def __init__(
        self,
        model: str = None,
        host: str = None,
        temperature: float = None
    ):
        """
        Initialize LLM client
        
        Args:
            model: Model name (e.g., "llama3:8b")
            host: Ollama host URL
            temperature: Sampling temperature (0=deterministic, 1=creative)
        """
        if ollama is None:
            raise ImportError(
                "ollama package not installed. Install with: pip install ollama"
            )
        
        self.model = model or settings.llm_model
        self.host = host or settings.ollama_host
        self.temperature = temperature or settings.llm_temperature
        
        # Configure ollama client
        self.client = ollama.Client(host=self.host)
        
        logger.info(f"LLM Client initialized")
        logger.info(f"  Model: {self.model}")
        logger.info(f"  Host: {self.host}")
        logger.info(f"  Temperature: {self.temperature}")
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Check if Ollama is running and model is available"""
        try:
            # List available models
            models_response = self.client.list()
            
            # Handle different response formats
            if isinstance(models_response, dict):
                models = models_response.get('models', [])
            else:
                models = models_response
            
            # Extract model names safely
            model_names = []
            for m in models:
                if isinstance(m, dict):
                    # Try different keys
                    name = m.get('name') or m.get('model') or str(m)
                    model_names.append(name)
                else:
                    model_names.append(str(m))
            
            logger.info(f"✅ Connected to Ollama")
            logger.info(f"   Available models: {model_names}")
            
            # Check if our model is available
            if self.model not in model_names:
                # Try without tag (e.g., "llama3" instead of "llama3:8b")
                base_model = self.model.split(':')[0]
                if not any(base_model in name for name in model_names):
                    logger.warning(
                        f"Model '{self.model}' not found. "
                        f"Pull it with: ollama pull {self.model}"
                    )
                else:
                    logger.info(f"✅ Model '{self.model}' is available")
            else:
                logger.info(f"✅ Model '{self.model}' is available")
                
        except Exception as e:
            logger.error(f"Failed to connect to Ollama: {e}")
            logger.error(f"Make sure Ollama is running at {self.host}")
            logger.error(f"Start it with: ollama serve")
            # Don't raise - allow the service to continue

    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: int = None,
        temperature: float = None
    ) -> str:
        """
        Generate a response from the LLM
        
        Args:
            prompt: The user prompt
            system_prompt: System instructions (optional)
            max_tokens: Max tokens to generate
            temperature: Override default temperature
            
        Returns:
            Generated text
        """
        temperature = temperature if temperature is not None else self.temperature
        max_tokens = max_tokens or settings.llm_max_tokens
        
        messages = []
        
        # Add system prompt if provided
        if system_prompt:
            messages.append({
                'role': 'system',
                'content': system_prompt
            })
        
        # Add user prompt
        messages.append({
            'role': 'user',
            'content': prompt
        })
        
        return self._generate_with_retry(messages, temperature, max_tokens)
    
    def _generate_with_retry(self, messages: List[Dict], temperature: float, max_tokens: int) -> str:
        """
        Internal method with retry logic for LLM generation
        Retries up to 3 times with exponential backoff on connection errors
        """
        max_attempts = 3
        base_wait = 2  # seconds
        
        for attempt in range(1, max_attempts + 1):
            try:
                logger.debug(f"Sending request to Ollama ({self.model})... (attempt {attempt}/{max_attempts})")
                
                response = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options={
                        'temperature': temperature,
                        'num_predict': max_tokens,
                    }
                )
                
                # Extract the response text
                answer = response['message']['content']
                
                logger.debug(f"✅ Got response ({len(answer)} chars)")
                
                return answer
                
            except (ConnectionError, TimeoutError) as e:
                # Retry on connection/timeout errors
                if attempt < max_attempts:
                    wait_time = base_wait * (2 ** (attempt - 1))  # Exponential backoff
                    logger.warning(f"LLM request failed (attempt {attempt}/{max_attempts}): {e}. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    logger.error(f"LLM generation failed after {max_attempts} attempts: {e}")
                    raise
            except Exception as e:
                # Don't retry on other errors (e.g., model not found)
                logger.error(f"LLM generation failed: {e}")
                raise
    
    def generate_with_context(
        self,
        query: str,
        context_chunks: List[LogChunk],
        system_prompt: Optional[str] = None,
        chat_context: Optional[str] = None  # Task 2: Optional chat history
    ) -> str:
        """
        Generate response using retrieved context (RAG)
        
        This is the core RAG method:
        1. Takes user query
        2. Takes relevant log chunks from vector search
        3. Formats them into a prompt
        4. Asks LLM to answer based on that context
        
        Args:
            query: User's question
            context_chunks: Relevant log chunks from search
            system_prompt: Custom system prompt (uses default if None)
            chat_context: Previous conversation for continuity
            
        Returns:
            LLM's answer
        """
        # Default system prompt for log analysis
        if system_prompt is None:
            system_prompt = """You are Sentry-AI, an expert system administrator and log analysis assistant.

Your job is to analyze log files and help diagnose system issues.

Guidelines:
- Be concise and technical
- Focus on root causes, not symptoms
- Cite specific log entries when making claims
- If you don't know, say so clearly
- Suggest next steps for investigation
- Use timestamps to establish sequence of events
"""
            # Add chat context instruction if we have history
            if chat_context:
                system_prompt += """
- If the user refers to previous conversation, use that context to provide relevant answers
"""
        
        # Build context from chunks
        context_text = self._format_context(context_chunks)
        
        # Build the full prompt with optional chat context
        prompt_parts = []
        
        # Add chat context if available (Task 2)
        if chat_context:
            prompt_parts.append(f"CONVERSATION HISTORY:\n{chat_context}\n")
        
        prompt_parts.append(f"""Based on the following log entries, answer this question:

QUESTION: {query}

LOG ENTRIES:
{context_text}

Provide a clear, actionable answer based ONLY on the log entries above. If the logs don't contain enough information to answer, say so.""")
        
        prompt = "\n".join(prompt_parts)
        
        return self.generate(prompt, system_prompt=system_prompt)

    
    def _format_context(self, chunks: List[LogChunk]) -> str:
        """
        Format log chunks into a readable context string
        
        Args:
            chunks: List of log chunks
            
        Returns:
            Formatted context string
        """
        if not chunks:
            return "(No log entries found)"
        
        context_lines = []
        
        for i, chunk in enumerate(chunks, 1):
            # Format timestamp
            timestamp = chunk.timestamp.strftime("%Y-%m-%d %H:%M:%S")
            
            # Format log level
            level = chunk.log_level.value.upper()
            
            # Get source info from metadata
            source_info = ""
            if 'file' in chunk.metadata:
                source_info = f" [{chunk.metadata['file']}]"
            elif 'event_id' in chunk.metadata:
                source_info = f" [EventID: {chunk.metadata['event_id']}]"
            
            # Build entry
            entry = f"""[{i}] {timestamp} | {level}{source_info}
{chunk.content}
"""
            context_lines.append(entry)
        
        return "\n".join(context_lines)
    
    
    def generate_context_summary(self, chunks: List[LogChunk], source_name: str = None) -> str:
        """
        Generate a contextual summary for a batch of log chunks
        
        This summary captures the overall context of the log file/source,
        which will be prepended to individual chunks during embedding.
        
        Args:
            chunks: List of log chunks from the same source
            source_name: Optional name of the source (file/eventlog)
            
        Returns:
            A concise summary describing the log context
        """
        if not chunks:
            return ""
        
        # Sample up to 10 chunks for context (to avoid token limits)
        sample_size = min(10, len(chunks))
        sample_chunks = chunks[:sample_size]
        
        # Build sample content
        sample_content = "\n".join([
            f"[{chunk.timestamp.strftime('%Y-%m-%d %H:%M:%S')}] [{chunk.log_level.value.upper()}] {chunk.content[:200]}"
            for chunk in sample_chunks
        ])
        
        # Extract metadata hints
        source_info = source_name or "Unknown Source"
        if chunks[0].metadata.get('file'):
            source_info = chunks[0].metadata['file']
        elif chunks[0].metadata.get('event_id'):
            source_info = f"Event Log (EventID: {chunks[0].metadata['event_id']})"
        
        # Build prompt for context generation
        prompt = f"""Analyze these log entries from "{source_info}" and provide a concise 2-3 sentence summary.

Focus on:
- What system/application these logs are from
- The general context or purpose
- Any notable patterns or themes

Sample log entries ({sample_size} of {len(chunks)} total):
{sample_content}

Provide ONLY the summary, no preamble."""
        
        try:
            # Generate summary with low temperature for consistency
            summary = self.generate(
                prompt,
                system_prompt="You are a log analysis expert. Provide concise, technical summaries.",
                temperature=0.3,
                max_tokens=150
            )
            
            # Clean up the summary
            summary = summary.strip()
            
            logger.debug(f"Generated context summary: {summary[:100]}...")
            return summary
            
        except Exception as e:
            logger.warning(f"Failed to generate context summary: {e}")
            # Fallback to simple metadata-based context
            return f"Log entries from {source_info} containing {len(chunks)} entries."

    def expand_query(self, query: str) -> str:
        """
        Expand user query using regex-based pattern matching
        
        This is faster and more deterministic than LLM-based expansion.
        Uses pattern matching to identify common query types and expand them
        with relevant technical terms found in logs.
        
        Args:
            query: User's original query
            
        Returns:
            Expanded query string with OR operators
        """
        import re
        
        # Define comprehensive expansion rules
        # Each pattern maps to a list of technical terms commonly found in logs
        expansion_rules = {
            # Performance & Latency Issues
            r'\b(slow|latency|lag|sluggish|delayed?|hang|hanging|hung|freeze|freezing|frozen)\b': [
                'high latency', 'slow response', 'slow response time', 'timeout', 
                'performance degradation', 'response time exceeded', 'request timeout',
                'slow query', 'delayed response', 'processing delay', 'lag detected',
                'performance issue', 'slow performance', 'latency spike', 'high response time'
            ],
            
            # Timeout Issues
            r'\b(timeout|timed?\s*out|time\s*limit)\b': [
                'timeout', 'connection timeout', 'read timeout', 'write timeout',
                'request timeout', 'operation timeout', 'execution timeout',
                'timeout exceeded', 'timeout error', 'timed out', 'deadline exceeded'
            ],
            
            # General Errors & Failures
            r'\b(error|fail(ure|ed|ing)?|crash(ed|ing)?|exception|bug|issue|problem|broken)\b': [
                'error', 'exception', 'failure', 'failed', 'crash', 'crashed',
                'fatal error', 'critical error', 'runtime error', 'system error',
                'application error', 'service error', 'unhandled exception',
                'stack trace', 'error code', 'error message', 'exception thrown'
            ],
            
            # Database Issues
            r'\b(database|db|sql|mysql|postgres|postgresql|mongo|mongodb|redis|oracle)\b': [
                'database error', 'database connection failed', 'connection refused',
                'deadlock', 'deadlock detected', 'query timeout', 'slow query',
                'connection pool exhausted', 'too many connections', 'max connections',
                'database unavailable', 'connection lost', 'connection timeout',
                'query failed', 'transaction failed', 'lock timeout', 'table lock',
                'duplicate key', 'foreign key constraint', 'constraint violation'
            ],
            
            # Connection Issues
            r'\b(connection|connect|socket|tcp|udp)\b': [
                'connection refused', 'connection failed', 'connection timeout',
                'connection reset', 'connection lost', 'connection closed',
                'connection error', 'socket error', 'socket timeout', 'socket closed',
                'unable to connect', 'connection aborted', 'connection dropped',
                'broken pipe', 'connection pool', 'max connections reached'
            ],
            
            # Network Issues
            r'\b(network|dns|host|unreachable|ping)\b': [
                'network error', 'network unreachable', 'host unreachable',
                'DNS resolution failed', 'DNS lookup failed', 'DNS error',
                'no route to host', 'network timeout', 'network connection failed',
                'unable to resolve', 'hostname not found', 'connection refused',
                'network is down', 'network unavailable', 'packet loss'
            ],
            
            # Memory Issues
            r'\b(memory|ram|heap|oom|out\s*of\s*memory|leak)\b': [
                'out of memory', 'OOM', 'OutOfMemoryError', 'memory leak',
                'heap exhausted', 'heap space', 'memory pressure', 'memory exceeded',
                'insufficient memory', 'memory allocation failed', 'memory limit',
                'heap overflow', 'stack overflow', 'memory error', 'GC overhead'
            ],
            
            # Disk & Storage Issues
            r'\b(disk|storage|filesystem|file\s*system|i/?o|partition|volume)\b': [
                'disk full', 'no space left', 'disk space', 'storage full',
                'I/O error', 'disk error', 'disk read error', 'disk write error',
                'filesystem error', 'file system full', 'quota exceeded',
                'disk failure', 'disk unavailable', 'read-only filesystem',
                'permission denied', 'access denied', 'file not found'
            ],
            
            # CPU Issues
            r'\b(cpu|processor|core|thread|process)\b': [
                'high CPU', 'CPU usage', 'CPU spike', 'CPU overload',
                'CPU throttling', 'processor overload', 'thread pool exhausted',
                'too many threads', 'process killed', 'process terminated',
                'deadlock', 'thread deadlock', 'thread blocked', 'thread timeout'
            ],
            
            # Authentication & Authorization
            r'\b(auth(entication)?|login|logout|sign\s*in|sign\s*out|credential|token|session)\b': [
                'authentication failed', 'authentication error', 'login failed',
                'invalid credentials', 'unauthorized', 'access denied',
                'permission denied', 'forbidden', 'token expired', 'token invalid',
                'session expired', 'session timeout', 'invalid token',
                'authentication required', 'credentials invalid', '401', '403'
            ],
            
            # Permission & Access Issues
            r'\b(permission|access|forbidden|denied|unauthorized)\b': [
                'permission denied', 'access denied', 'forbidden', 'unauthorized',
                'insufficient permissions', 'access forbidden', 'not authorized',
                '403', '401', 'authentication required', 'authorization failed'
            ],
            
            # HTTP Status Codes & API Errors
            r'\b(4\d{2}|5\d{2}|http|api|rest|endpoint|request|response)\b': [
                '400', '401', '403', '404', '405', '408', '429', '500', '502', '503', '504',
                'bad request', 'unauthorized', 'forbidden', 'not found', 'method not allowed',
                'request timeout', 'too many requests', 'internal server error',
                'bad gateway', 'service unavailable', 'gateway timeout',
                'API error', 'endpoint error', 'HTTP error', 'status code'
            ],
            
            # Service Unavailability
            r'\b(unavailable|down|offline|outage|maintenance)\b': [
                'service unavailable', 'service down', 'service offline',
                'system unavailable', 'system down', 'outage', 'downtime',
                'maintenance mode', 'temporarily unavailable', '503',
                'service not available', 'service unreachable'
            ],
            
            # Deployment & Configuration
            r'\b(deploy(ment)?|config(uration)?|setup|install|update|upgrade|migration)\b': [
                'deployment failed', 'deployment error', 'configuration error',
                'config error', 'misconfiguration', 'setup failed', 'installation failed',
                'update failed', 'upgrade failed', 'migration failed', 'rollback',
                'invalid configuration', 'configuration missing', 'setup error'
            ],
            
            # Service-Specific: Checkout/Payment
            r'\b(checkout|payment|cart|order|transaction|purchase)\b': [
                'checkout failed', 'checkout error', 'payment failed', 'payment error',
                'payment declined', 'transaction failed', 'transaction error',
                'order failed', 'order error', 'cart error', 'purchase failed',
                'payment gateway error', 'payment timeout', 'transaction timeout'
            ],
            
            # Service-Specific: Email
            r'\b(email|mail|smtp|send|notification)\b': [
                'email failed', 'email error', 'SMTP error', 'send failed',
                'mail delivery failed', 'notification failed', 'email timeout',
                'SMTP connection failed', 'mail server error', 'delivery error'
            ],
            
            # Service-Specific: Upload/Download
            r'\b(upload|download|file\s*transfer|ftp|s3|blob|object\s*storage)\b': [
                'upload failed', 'download failed', 'file transfer failed',
                'upload error', 'download error', 'transfer error', 'FTP error',
                'S3 error', 'storage error', 'blob error', 'file upload timeout',
                'file download timeout', 'transfer timeout'
            ],
            
            # Validation & Data Issues
            r'\b(validat(ion|e)|invalid|corrupt(ed)?|malformed|parse|parsing)\b': [
                'validation failed', 'validation error', 'invalid data',
                'invalid input', 'invalid format', 'data corruption', 'corrupted data',
                'malformed data', 'malformed request', 'parse error', 'parsing failed',
                'invalid JSON', 'invalid XML', 'schema validation failed'
            ],
            
            # Security Issues
            r'\b(security|vulnerability|breach|attack|injection|xss|csrf|malicious)\b': [
                'security error', 'security violation', 'vulnerability detected',
                'security breach', 'attack detected', 'SQL injection', 'XSS attack',
                'CSRF attack', 'malicious request', 'suspicious activity',
                'unauthorized access', 'intrusion detected', 'security alert'
            ],
            
            # Rate Limiting & Throttling
            r'\b(rate\s*limit|throttl(e|ing)|quota|limit\s*exceed)\b': [
                'rate limit exceeded', 'rate limited', 'throttled', 'throttling',
                'quota exceeded', 'limit exceeded', 'too many requests', '429',
                'request limit', 'API limit', 'usage limit exceeded'
            ],
            
            # Retry & Backoff
            r'\b(retry|retrying|backoff|attempt)\b': [
                'retry failed', 'max retries exceeded', 'retry limit',
                'retrying request', 'exponential backoff', 'retry attempt',
                'all retries failed', 'retry exhausted'
            ],
            
            # Cache Issues
            r'\b(cache|caching|redis|memcache)\b': [
                'cache error', 'cache miss', 'cache expired', 'cache unavailable',
                'Redis error', 'Redis connection failed', 'cache timeout',
                'cache invalidation', 'cache write failed', 'cache read failed'
            ],
            
            # Queue & Message Issues
            r'\b(queue|message|kafka|rabbitmq|sqs|pubsub|topic)\b': [
                'queue error', 'message failed', 'queue full', 'message timeout',
                'Kafka error', 'RabbitMQ error', 'SQS error', 'message delivery failed',
                'queue overflow', 'message lost', 'consumer error', 'producer error',
                'topic error', 'subscription error', 'message processing failed'
            ],
            
            # Load Balancer & Proxy
            r'\b(load\s*balanc(er|ing)|proxy|nginx|haproxy|gateway)\b': [
                'load balancer error', 'proxy error', 'gateway error', 'upstream error',
                'backend unavailable', 'no healthy upstream', 'proxy timeout',
                'gateway timeout', '502', '504', 'upstream timeout',
                'load balancing failed', 'proxy connection failed'
            ],
            
            # Container & Orchestration
            r'\b(docker|container|kubernetes|k8s|pod|node|cluster)\b': [
                'container failed', 'container error', 'pod failed', 'pod error',
                'node unavailable', 'node failure', 'cluster error', 'deployment failed',
                'container crashed', 'pod crashed', 'OOMKilled', 'CrashLoopBackOff',
                'ImagePullBackOff', 'container restart', 'pod restart'
            ],
            
            # Logging & Monitoring
            r'\b(log|logging|monitor|metric|alert|warning|critical)\b': [
                'log error', 'logging failed', 'warning', 'critical', 'alert',
                'monitoring error', 'metric error', 'log write failed',
                'log rotation failed', 'alert triggered', 'threshold exceeded'
            ],
        }
        
        # Collect expansions based on pattern matches
        expansions = set()
        query_lower = query.lower()
        
        for pattern, terms in expansion_rules.items():
            if re.search(pattern, query_lower, re.IGNORECASE):
                expansions.update(terms)
        
        # If we found expansions, combine with original query
        if expansions:
            # Include original query + all matched expansion terms
            all_terms = [query] + sorted(list(expansions))
            expanded = ' OR '.join(all_terms)
            logger.info(f"Expanded query: '{query}' -> {len(expansions)} terms added")
            logger.debug(f"Expansion preview: {' OR '.join(list(expansions)[:5])}...")
            return expanded
        
        # No patterns matched, return original query
        logger.debug(f"No expansion patterns matched for: '{query}'")
        return query

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = None
    ) -> str:
        """
        Multi-turn conversation
        
        Args:
            messages: List of message dicts with 'role' and 'content'
                     Example: [
                         {'role': 'user', 'content': 'What errors?'},
                         {'role': 'assistant', 'content': 'Found 3 errors...'},
                         {'role': 'user', 'content': 'Show me the first one'}
                     ]
            temperature: Override default temperature
            
        Returns:
            Assistant's response
        """
        temperature = temperature if temperature is not None else self.temperature
        
        try:
            response = self.client.chat(
                model=self.model,
                messages=messages,
                options={'temperature': temperature}
            )
            
            return response['message']['content']
            
        except Exception as e:
            logger.error(f"Chat failed: {e}")
            raise
    
    def stream_generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None
    ):
        """
        Stream response token by token (for UI responsiveness)
        
        Yields:
            Chunks of text as they're generated
        """
        messages = []
        
        if system_prompt:
            messages.append({'role': 'system', 'content': system_prompt})
        
        messages.append({'role': 'user', 'content': prompt})
        
        try:
            stream = self.client.chat(
                model=self.model,
                messages=messages,
                stream=True,
                options={'temperature': self.temperature}
            )
            
            for chunk in stream:
                if 'message' in chunk and 'content' in chunk['message']:
                    yield chunk['message']['content']
                    
        except Exception as e:
            logger.error(f"Streaming failed: {e}")
            raise
    
    def get_model_info(self) -> dict:
        """Get information about the current model"""
        try:
            # Show model details
            info = self.client.show(self.model)
            return {
                'model': self.model,
                'host': self.host,
                'temperature': self.temperature,
                'details': info
            }
        except Exception as e:
            logger.error(f"Failed to get model info: {e}")
            return {
                'model': self.model,
                'host': self.host,
                'temperature': self.temperature,
                'error': str(e)
            }

    def list_models(self) -> List[str]:
        """
        List available Ollama models
        
        Returns:
            List of model names (e.g. ["llama3:8b", "mistral"])
        """
        try:
            # List available models
            models_response = self.client.list()
            
            # Handle different response formats
            if isinstance(models_response, dict):
                models = models_response.get('models', [])
            else:
                models = models_response
            
            # Extract model names safely
            model_names = []
            for m in models:
                if isinstance(m, dict):
                    # Try different keys
                    name = m.get('name') or m.get('model') or str(m)
                    model_names.append(name)
                else:
                    model_names.append(str(m))
            
            return sorted(model_names)
            
        except Exception as e:
            logger.error(f"Failed to list models: {e}")
            return []
    
    def __repr__(self):
        return f"<LLMClient(model={self.model}, host={self.host})>"


# ===== SINGLETON INSTANCE =====
_llm_client = None


def get_llm_client() -> LLMClient:
    """
    Get the global LLM client instance
    Lazy-loads on first call
    """
    global _llm_client
    if _llm_client is None:
        _llm_client = LLMClient()
    return _llm_client