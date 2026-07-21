
import os
from typing import List, Dict
from dotenv import load_dotenv
from equiaudit.models.clients.base_client import BaseModelClient, ModelInfo

try:
    from google import genai
    from google.genai import types
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False
    genai = None
    types = None

load_dotenv()


class GeminiClient(BaseModelClient):
    """
    Client for Google Gemini API using the google-genai SDK.

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
                "google-genai package not installed. "
                "Install with: pip install google-genai"
            )

        self.model_name = model
        self.api_key = api_key or os.getenv("GOOGLE_API_KEY")

        if not self.api_key:
            raise ValueError(
                "Google API key not found. "
                "Set GOOGLE_API_KEY environment variable or pass api_key parameter."
            )

        self.client = genai.Client(api_key=self.api_key)
        self._initialized = True

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        self.validate_messages(messages)

        system_instruction = None
        history = []

        for msg in messages:
            role = msg["role"]
            content = msg["content"]

            if role == "system":
                system_instruction = content
            elif role == "user":
                history.append(types.Content(role="user", parts=[types.Part(text=content)]))
            elif role == "assistant":
                history.append(types.Content(role="model", parts=[types.Part(text=content)]))

        # Split off the last user message — it is sent via chat.send_message
        last_message = ""
        if history and history[-1].role == "user":
            last_message = history.pop().parts[0].text

        config = types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
            system_instruction=system_instruction,
        )

        try:
            chat = self.client.chats.create(
                model=self.model_name,
                history=history,
                config=config,
            )
            response = chat.send_message(last_message)
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
