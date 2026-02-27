from typing import List, Dict
from models.clients.base_client import BaseModelClient, ModelInfo

try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    AutoModelForCausalLM = None
    AutoTokenizer = None


class LocalModelClient(BaseModelClient):
    """
    Client for locally-hosted models using HuggingFace Transformers.
    """
        
    def __init__(
        self, 
        model: str = None,
        device: str = "auto",
        dtype: str = "float16",
        **kwargs
    ):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError(
                "PyTorch and transformers are required for local model inference. "
                "Install them with: pip install torch transformers\n"
                "Alternatively, use API-based models (Gemini or OpenRouter) which don't require PyTorch."
            )
        
        self.model_name = model
        self.device = device
        
        dtype_map = {
            "float16": torch.float16,
            "float32": torch.float32,
            "bfloat16": torch.bfloat16
        }
        self.dtype = dtype_map.get(dtype, torch.float16)
        
        # Load tokenizer and model
        print(f"Loading local model: {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=self.dtype,
            device_map=self.device
        )
        
        self._initialized = True
        print(f"Local model loaded: {self.model_name}")
    
    def generate(
        self, 
        messages: List[Dict[str, str]], 
        temperature: float = 0.2, 
        max_tokens: int = 4096,
        **kwargs
    ) -> str:
        self.validate_messages(messages)
        
        prompt = self._build_prompt_from_messages(messages)
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)
        
        # Generate
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=max(temperature, 0.01),  
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=kwargs.get("num_beams", 1),
                top_p=kwargs.get("top_p", 0.9),
                repetition_penalty=kwargs.get("repetition_penalty", 1.1)
            )
        
        text = self.tokenizer.decode(output[0], skip_special_tokens=True)  
        response = text[len(prompt):].strip()
        
        return response
    
    def get_model_info(self) -> ModelInfo:
        name_lower = self.model_name.lower()
        
        supports_function = any(
            keyword in name_lower 
            for keyword in ["function", "granite", "code", "instruct"]
        )
        
        return ModelInfo(
            name=self.model_name,
            provider="local",
            supports_vision=False,
            supports_function_calling=supports_function,
            supports_json_output=True,
            supports_structured_output=supports_function,
            supports_streaming=False,
            max_tokens=4096,
            context_window=8192
        )
