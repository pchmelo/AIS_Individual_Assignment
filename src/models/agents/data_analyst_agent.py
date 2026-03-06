import json
from models.agents.base_agent import BaseAgent
from models.clients.base_client import BaseModelClient
from tools.tool_manager import ToolManager

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

                    OUTPUT FORMATTING RULES (MUST FOLLOW):
                    - Use ## for main section headers (e.g., ## Summary)
                    - Use ### for subsection headers (e.g., ### Key Findings)
                    - Use numbered lists (1. 2. 3.) for ordered items
                    - Use bullet points (- item) for unordered lists
                    - For sensitive attribute tables, use this EXACT format:
                      1. Column: ColumnName | Reason: Description | Values: [val1, val2, val3]
                      (NO bold markers ** around Column:)
                    - Avoid using ** bold markers ** in headers - just use the markdown header syntax
                    - Keep text clean without excessive formatting
                    - DO NOT use emojis, icons, or special symbols (no ✓, ✗, ■, ●, etc.)
                """
    
    def run(self, user_message: str, max_retries: int = 3) -> str:
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_message}
        ]
        
        # Retry logic for model failures
        model_reply = None
        for attempt in range(max_retries):
            model_reply = self.ask_model(messages, max_tokens=4096)
            if model_reply is not None:
                break
            print(f"Model attempt {attempt + 1}/{max_retries} failed, retrying...")
        
        if model_reply is None:
            return "Error: Model returned no response after multiple attempts. Please check your API connection and try again."
        
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
                "content": f"""
                    TOOL EXECUTION RESULT:
                    {json.dumps(result, indent=2)}
                    Provide a comprehensive analysis including:
                    1. Summary of key findings from the data
                    2. Specific issues identified with numbers
                    3. Severity assessment
                    4. Recommendations
                """
            })
            
            final_response = self.ask_model(messages, temperature=0.3, max_tokens=4096)
            print(f"Analysis complete: {len(final_response)} chars")
            return final_response
            
        except ValueError as e:
            return f"Error: {str(e)}"