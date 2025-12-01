from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import requests
import os
from dotenv import load_dotenv

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