from agents.base_agent import BaseAgent
from agents.model_client import BaseModelClient
from tools.tool_manager import ToolManager
import json

class DataAnalystAgent(BaseAgent):    
    def __init__(self, tool_manager: ToolManager, model_client: BaseModelClient = None, model_name: str = None):
        super().__init__(model_client=model_client, model_name=model_name)
        self.tool_manager = tool_manager
    
    def get_system_prompt(self) -> str:
        return f"""You are an expert data analyst assistant.
                    You help users understand datasets, identify patterns, and provide actionable insights.

                    Available analysis tools: {self.tool_manager.get_tool_descriptions_json()}

                    When analyzing data:
                    - Identify key patterns and trends
                    - Highlight potential issues (missing data, imbalances, biases)
                    - Provide recommendations
                    - Use precise numbers from tool outputs
                """
    
    def run(self, user_message: str) -> str:
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_message}
        ]
        
        model_reply = self.ask_model(messages, max_tokens=512)
        
        try:
            tool_name, args = self.tool_manager.parse_function_call(model_reply)
            
            if tool_name is None:
                return model_reply
            
            result = self.tool_manager.execute_tool(tool_name, args)
            print(f"Tool '{tool_name}' executed - Result type: {type(result).__name__}")
            
            if isinstance(result, dict) and result.get("status") == "error":
                return json.dumps(result, indent=2)
            
            messages.append({"role": "assistant", "content": model_reply})
            messages.append({
                "role": "system",
                "content": f"""TOOL EXECUTION RESULT:
{json.dumps(result, indent=2)}

Provide a comprehensive analysis including:
1. Summary of key findings from the data
2. Specific issues identified with numbers
3. Severity assessment
4. Recommendations"""
            })
            
            final_response = self.ask_model(messages, temperature=0.3, max_tokens=4096)
            print(f"Analysis complete: {len(final_response)} chars")
            return final_response
            
        except ValueError as e:
            return f"Error: {str(e)}"