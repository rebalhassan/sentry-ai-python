# sentry/services/llm.py
"""
LLM client for talking to Ollama
Generates natural language responses based on retrieved context
"""

import logging
from typing import List, Dict, Optional
import json

try:
    import ollama
except ImportError:
    ollama = None

from ..core.config import settings
from ..core.models import LogChunk

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
        
        try:
            logger.debug(f"Sending request to Ollama ({self.model})...")
            logger.debug(f"  Prompt length: {len(prompt)} chars")
            
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
            
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            raise
    
    def generate_with_context(
        self,
        query: str,
        context_chunks: List[LogChunk],
        system_prompt: Optional[str] = None
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
        
        # Build context from chunks
        context_text = self._format_context(context_chunks)
        
        # Build the full prompt
        prompt = f"""Based on the following log entries, answer this question:

QUESTION: {query}

LOG ENTRIES:
{context_text}

Provide a clear, actionable answer based ONLY on the log entries above. If the logs don't contain enough information to answer, say so."""
        
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