
import os
from typing import List, Dict
from dotenv import load_dotenv
from models.clients.base_client import BaseModelClient, ModelInfo

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None

load_dotenv()


class GeminiClient(BaseModelClient):
    """
    Client for Google Gemini API.
    
    Environment Variables:
        GOOGLE_API_KEY: Your Google AI API key (required)
    """
        
    def __init__(
        self, 
        model: str = None,
        api_key: str = None,
        **kwargs
    ):
        super().__init__()
        
        if not GEMINI_AVAILABLE:
            raise ImportError(
                "google-generativeai package not installed. "
                "Install with: pip install google-generativeai"
            )
        
        self.model_name = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError(
                "Google API key not found. "
                "Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)
        
        self._initialized = True
        print(f"Gemini client initialized: {self.model_name}")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2, 
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        self.validate_messages(messages)
        
        # Convert messages to Gemini format
        chat_history = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                system_instruction = content
            elif role == "user":
                chat_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                chat_history.append({"role": "model", "parts": [content]})
        
        # Configure generation parameters
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        try:
            # Create model with system instruction if provided
            if system_instruction:
                model_with_system = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
                chat = model_with_system.start_chat(
                    history=chat_history[:-1] if chat_history else []
                )
            else:
                chat = self.model.start_chat(
                    history=chat_history[:-1] if chat_history else []
                )
            
            # Send last message
            last_message = chat_history[-1]["parts"][0] if chat_history else ""
            response = chat.send_message(last_message, generation_config=generation_config)
            
            return response.text
            
        except Exception as e:
            raise Exception(f"Gemini API Error: {str(e)}")
    
    def get_model_info(self) -> ModelInfo:
        is_pro = "pro" in self.model_name.lower()
        is_vision = "vision" in self.model_name.lower()
        
        return ModelInfo(
            name=self.model_name,
            provider="google",
            supports_vision=is_vision or is_pro,
            supports_function_calling=True,
            supports_json_output=True,
            supports_structured_output=True,
            supports_streaming=True,
            max_tokens=8192 if is_pro else 4096,
            context_window=1000000 if "1.5" in self.model_name else 128000
        )
