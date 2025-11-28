from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
import json
from tools.tool_manager import ToolManager

class Agent:
    def __init__(self, tool_manager: ToolManager):
        self.model_name = "ibm-granite/granite-3b-code-instruct"
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            dtype=torch.float16,
            device_map="auto"
        )

        self.tool_manager = tool_manager

    def ask_model(self, messages):
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

        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.model.device)

        with torch.no_grad():  
            output = self.model.generate(
                **inputs,
                max_new_tokens=128,  
                temperature=0.2,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
                num_beams=1
            )

        text = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return text[len(prompt):].strip()


    def run_agent(self, user_message):
        messages = [
            {
                "role": "system", 
                "content": f"You are a helpful assistant. You can use tools via <functioncall> JSON. Available tools: {self.tool_manager.get_tool_descriptions_json()}"
            },
            {"role": "user", "content": user_message}
        ]

        model_reply = self.ask_model(messages)
        print("MODEL RAW OUTPUT:", model_reply)

        try:
            tool_name, args = self.tool_manager.parse_function_call(model_reply)
            
            if tool_name is None:
                return model_reply
            
            result = self.tool_manager.execute_tool(tool_name, args)
            
            messages.append({"role": "assistant", "content": model_reply})
            messages.append({
                "role": "system", 
                "content": f"""FUNCTION RESPONSE: {json.dumps(result)}

                IMPORTANT: You must use the EXACT values from the function response above. 
                Do not make up or approximate numbers. 
                Report the temperature as: {result.get('temperature', 'N/A')}{result.get('unit', '')}

                Now provide a natural language response using ONLY the data provided."""
            })
            
            final_response = self.ask_model(messages)
            print("FINAL RESPONSE:", final_response)
            return final_response
            
        except ValueError as e:
            return str(e)
