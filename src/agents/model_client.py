from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import os
from dotenv import load_dotenv

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

load_dotenv()

class BaseModelClient(ABC):    
    @abstractmethod
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 2048) -> str:
        pass
    
    @abstractmethod
    def supports_function_calling(self) -> bool:
        pass


class LocalModelClient(BaseModelClient):    
    def __init__(self, model_name: str = "ibm-granite/granite-3b-code-instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto"
        )
        print(f"Local model loaded: {model_name}")
    
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 2048) -> str:
        prompt = self._build_prompt(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt):].strip()
    
    def _build_prompt(self, messages: List[Dict[str, str]]) -> str:
        prompt = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                prompt += f"System: {content}\n"
            elif role == "user":
                prompt += f"User: {content}\n"
            elif role == "assistant":
                prompt += f"Assistant: {content}\n"
        
        prompt += "Assistant:"
        return prompt
    
    def supports_function_calling(self) -> bool:
        return "function" in self.model_name.lower() or "granite" in self.model_name.lower()


class OpenRouterClient(BaseModelClient):    
    def __init__(
        self, 
        model: str = "x-ai/grok-4.1-fast:free",
        base_url: str = "https://openrouter.ai/api/v1",
        model_info: Dict[str, Any] = None
    ):
        self.model = model
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        self.base_url = base_url
        self.model_info = model_info or {
            "vision": False,
            "function_calling": True,
            "json_output": False,
            "family": "unknown",
            "structured_output": True,
            "reasoning": True
        }
        
        if not self.api_key:
            raise ValueError("OpenRouter API key not found. Set OPENROUTER_API_KEY environment variable.")
        
        print(f"OpenRouter client initialized: {model}")
    
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 2048) -> str:
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "HTTP-Referer": "http://localhost:3000",
            "X-Title": "AI Agent" 
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        response = requests.post(
            f"{self.base_url}/chat/completions",
            headers=headers,
            json=payload
        )
        
        if response.status_code != 200:
            raise Exception(f"API Error: {response.status_code} - {response.text}")
        
        result = response.json()
        return result["choices"][0]["message"]["content"]
    
    def supports_function_calling(self) -> bool:
        return self.model_info.get("function_calling", False)


class GeminiClient(BaseModelClient):    
    def __init__(self, model: str = "gemini-3-pro-preview"):
        if not GEMINI_AVAILABLE:
            raise ImportError("google-generativeai package not installed. Install with: pip install google-generativeai")
        
        self.model_name = model
        self.api_key = os.getenv("GOOGLE_API_KEY")
        
        if not self.api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(model)
        
        print(f"Gemini client initialized: {model}")
    
    def generate(self, messages: List[Dict[str, str]], temperature: float = 0.2, max_tokens: int = 2048) -> str:        
        chat_history = []
        system_instruction = None
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                # Gemini uses system_instruction separately
                system_instruction = content
            elif role == "user":
                chat_history.append({"role": "user", "parts": [content]})
            elif role == "assistant":
                chat_history.append({"role": "model", "parts": [content]})
        
        generation_config = genai.types.GenerationConfig(
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        
        try:
            if system_instruction:
                model_with_system = genai.GenerativeModel(
                    self.model_name,
                    system_instruction=system_instruction
                )
                chat = model_with_system.start_chat(history=chat_history[:-1] if chat_history else [])
            else:
                chat = self.model.start_chat(history=chat_history[:-1] if chat_history else [])
            
            last_message = chat_history[-1]["parts"][0] if chat_history else ""
            response = chat.send_message(last_message, generation_config=generation_config)
            
            return response.text
        
        except Exception as e:
            raise Exception(f"Gemini API Error: {str(e)}")
    
    def supports_function_calling(self) -> bool:
        return True