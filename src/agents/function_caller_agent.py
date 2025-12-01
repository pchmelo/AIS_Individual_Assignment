import json
from agents.base_agent import BaseAgent
from agents.model_client import BaseModelClient
from tools.tool_manager import ToolManager

class FunctionCallerAgent(BaseAgent):
    def __init__(
        self, 
        tool_manager: ToolManager,
        model_client: BaseModelClient = None,
        model_name: str = None,
        reflect_on_tool_use: bool = True
    ):
        super().__init__(model_client=model_client, model_name=model_name)
        self.tool_manager = tool_manager
        self.reflect_on_tool_use = reflect_on_tool_use
    
    def get_system_prompt(self) -> str:
        return f"""You are a helpful assistant specialized in using tools to help users.
                   You can use tools via <functioncall> JSON format.
   
                   Available tools: {self.tool_manager.get_tool_descriptions_json()}
   
                   IMPORTANT: 
                   - Only output the <functioncall> tag with the tool name and arguments
                   - Do NOT generate the function response yourself
                   - The system will execute the tool and provide the result
                   - After receiving the result, provide a natural language response
   
                   Example:
                   User: What's the temperature in Porto?
                   Assistant: <functioncall> {{"name": "current_temp", "arguments": {{"city": "Porto"}}}}
                """
    
    def run(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_message}
        ]

        model_reply = self.ask_model(messages)
        model_reply = model_reply.split("FUNCTION RESPONSE:")[0].strip()
        
        try:
            tool_name, args = self.tool_manager.parse_function_call(model_reply)
            
            if tool_name is None:
                return model_reply
            
            result = self.tool_manager.execute_tool(tool_name, args)
            print(f"Tool '{tool_name}' executed - Result type: {type(result).__name__}")
            
            messages.append({
                "role": "assistant", 
                "content": f"<functioncall> {json.dumps({'name': tool_name, 'arguments': args})}"
            })
            
            if self.reflect_on_tool_use:
                messages.append({
                    "role": "system", 
                    "content": f"""FUNCTION RESPONSE: {json.dumps(result)}

                    Provide a natural language response using the EXACT values from the function response above.
                    Do not make up or approximate any numbers."""
                })
                
                final_response = self.ask_model(messages, temperature=0.1)
                print(f"Final response: {len(final_response)} chars")
                return final_response
            else:
                return json.dumps(result, indent=2)
            
        except ValueError as e:
            return f"Error: {str(e)}"