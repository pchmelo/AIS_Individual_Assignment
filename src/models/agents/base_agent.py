from abc import ABC, abstractmethod
from typing import List, Dict
import time
import re
from models.clients.base_client import BaseModelClient
from models.clients.local_client import LocalModelClient


# Patterns that indicate a model is outputting internal reasoning/monologue
# rather than the actual response content.
_REASONING_PREAMBLE_PATTERNS = re.compile(
    r'^(?:'
    r'Alright,\s*I need to|'
    r'Alright,\s*let me|'
    r'Okay,\s*let me|'
    r'Okay,\s*I need to|'
    r'Let me start by|'
    r'Let me think|'
    r'First,\s*I need to|'
    r'First,\s*let me|'
    r'I need to produce|'
    r'I need to analyze|'
    r'I need to review|'
    r'The user wants me to|'
    r'The user is asking|'
    r'Looking at the data|'
    r'Now I need to|'
    r'Now,\s+I need to|'
    r'Now,\s+let me'
    r')',
    re.IGNORECASE | re.MULTILINE
)

# Inline reasoning: a paragraph that IS reasoning (not report content).
# These lines start with first-person reasoning phrases.
_INLINE_REASONING_LINE_PATTERN = re.compile(
    r'^(?:'
    r'(?:Alright|Okay|Now)[,.]?\s+(?:I|let me|we need)|'
    r'Let me (?:start|think|extract|now|draft|compute|structure|analyze|check|see|look|write|calculate|plan)|'
    r'(?:I|We)(?:\'ll| will| should| need to| must| can| have to)\s+(?:start|think|extract|draft|compute|structure|analyze|check|see|look|write|calculate|plan|produce|include|present|use|assume|skip|omit|note|mention)|'
    r'(?:I|We) (?:need|think|believe|realize|notice|note|see|found|know|have|want|also|should)|'
    r'Note(?:\s+that)?:|'
    r'Wait,|'
    r'Actually,|'
    r'But (?:I|wait|note|the|we)|'
    r'Hmm,|'
    r'So (?:I|my|for each|now)|'
    r'For example,\s+(?:I|we)|'
    r'This (?:is|means|would|will|can|might|may) (?:a|the|my|our|an|be|help)|'
    r'Looking at|'
    r'From the (?:user|data|analysis|report)|'
    r'The user (?:said|wants|asked|expects|provided|gave)|'
    r'Since (?:I|we|the user)|'
    r'However,\s+(?:I|the user|we)|'
    r'Based on (?:the|this|my)'
    r')',
    re.IGNORECASE | re.MULTILINE
)


class APIError(Exception):
    """Exception raised for API errors with user-friendly messages."""
    
    ERROR_MESSAGES = {
        401: "Authentication failed. Check your API key is correct and active.",
        403: "Access forbidden. Your API key may not have permission for this model.",
        429: "Rate limit exceeded. Please wait and try again.",
        500: "Server error. The API provider is experiencing issues.",
        502: "Bad gateway. The API provider is temporarily unavailable.",
        503: "Service unavailable. The API provider is overloaded.",
    }
    
    def __init__(self, message: str, status_code: int = None):
        self.status_code = status_code
        self.original_message = message
        
        # Parse status code from message if not provided
        if status_code is None:
            match = re.search(r'(\d{3})', message)
            if match:
                status_code = int(match.group(1))
                self.status_code = status_code
        
        # Get user-friendly message
        friendly = self.ERROR_MESSAGES.get(status_code, "")
        if friendly:
            super().__init__(f"{friendly}")
        else:
            super().__init__(message)

    def __str__(self):
        return super().__str__()

class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    """
    def __init__(
        self, 
        model_client: BaseModelClient = None, 
        model_name: str = None
    ):
        if model_client is None:
            model_name = model_name
            self.model_client = LocalModelClient(model=model_name)
        else:
            self.model_client = model_client
    
    def ask_model(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2, 
        max_tokens: int = 4096,
        max_retries: int = 3
    ) -> str:
        """
        Send messages to the model and get a response.
        Raises APIError if the model fails to respond after retries.
        """
        last_error = None
        for attempt in range(max_retries):
            try:
                result = self.model_client.generate(messages, temperature, max_tokens)
                if result:
                    # Strip explicit reasoning blocks (e.g. DeepSeek's <think> tags)
                    result = re.sub(r'<think>.*?</think>', '', result, flags=re.DOTALL)
                    # Strip implicit reasoning leakage (model "thinking out loud")
                    result = self._strip_reasoning_leakage(result)
                    return result.strip()
            except Exception as e:
                last_error = e
                error_str = str(e)
                
                # Check for non-retryable errors (auth issues)
                if "401" in error_str or "403" in error_str:
                    raise APIError(error_str)
            
            if attempt < max_retries - 1:
                delay = 2 ** attempt
                time.sleep(delay)
        
        # All retries exhausted
        if last_error:
            raise APIError(str(last_error))
        raise APIError("Model returned empty response after all retries.")
    
    def _strip_reasoning_leakage(self, text: str) -> str:
        """
        Strip internal monologue/reasoning that leaks into the response when
        a 'thinking' model doesn't properly wrap its thoughts in tags.
        
        Two strategies applied:
        1. If the text STARTS with a reasoning preamble, find the first
           Markdown header and discard everything before it.
        2. Strip inline reasoning paragraphs (model 'thinking out loud' between
           sections of the actual report).
        """
        if not text:
            return text
        
        stripped = text.strip()
        
        # Strategy 1: Strip leading preamble
        if _REASONING_PREAMBLE_PATTERNS.match(stripped):
            # Find the first markdown header (##, ###, # followed by space)
            header_match = re.search(r'^#{1,4} ', stripped, re.MULTILINE)
            if header_match:
                stripped = stripped[header_match.start():].strip()
            else:
                # No header found - try to find a separator like --- or a bold line
                sep_match = re.search(r'^---$|^\*\*[^\n]+\*\*$', stripped, re.MULTILINE)
                if sep_match:
                    stripped = stripped[sep_match.start():].strip()
        
        # Strategy 2: Strip inline reasoning paragraphs
        stripped = self._strip_inline_reasoning(stripped)
        
        return stripped
    
    def _strip_inline_reasoning(self, text: str) -> str:
        """
        Remove paragraphs that look like internal model reasoning from a
        multi-paragraph response. A reasoning paragraph is identified when
        the majority of its lines match reasoning phrase patterns.
        """
        if not text:
            return text
        
        # Split into paragraphs (separated by blank lines)
        paragraphs = re.split(r'\n{2,}', text)
        clean_paragraphs = []
        
        for para in paragraphs:
            lines = [l for l in para.strip().splitlines() if l.strip()]
            if not lines:
                continue
            
            # Count lines that match reasoning patterns
            reasoning_lines = sum(
                1 for line in lines
                if _INLINE_REASONING_LINE_PATTERN.match(line.strip())
            )
            
            # If MORE THAN HALF the lines look like reasoning AND the paragraph
            # doesn't start with a Markdown header or table — drop it.
            is_header = lines[0].strip().startswith('#')
            is_table = lines[0].strip().startswith('|')
            is_bullet = lines[0].strip().startswith(('-', '*', '+', '1', '2', '3', '4', '5'))
            
            if reasoning_lines > len(lines) * 0.5 and not is_header and not is_table and not is_bullet:
                continue  # Skip this reasoning paragraph
            
            clean_paragraphs.append(para.strip())
        
        return '\n\n'.join(clean_paragraphs)
    
    @abstractmethod
    def run(self, user_message: str) -> str:
        """
        Process a user message and return a response.
        """
        pass
    
    @abstractmethod
    def get_system_prompt(self) -> str:
        """
        Get the system prompt for this agent.
        """
        pass
