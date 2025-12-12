import json
import re

class ToolManager:
    def __init__(self, tools: list = None):
        self.list_of_tools = tools if tools is not None else []
        self._build_tool_mappings()
    
    def _build_tool_mappings(self):
        self.tools = {tool.name: tool.function for tool in self.list_of_tools}
        self.tool_descriptions = [
            {
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters
            }
            for tool in self.list_of_tools
        ]
    
    def get_tool_descriptions_json(self):
        return json.dumps(self.tool_descriptions)
    
    def parse_function_call(self, model_output: str):
        tool_match = re.search(r"<functioncall>\s*({.*?})\s*(?:\n|$)", model_output, re.DOTALL)
        
        if tool_match:
            json_str = tool_match.group(1).strip()
        else:
            json_match = re.search(r'{\s*"name"\s*:\s*"[^"]+"\s*,\s*"(?:parameters|arguments)"\s*:\s*{[^}]*}\s*}', model_output, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                return None, None
        
        try:
            print(f"DEBUG - Extracted JSON: {json_str[:200]}...")
            
            tool_json = json.loads(json_str)
            tool_name = tool_json["name"]
            
            args = tool_json.get("arguments") or tool_json.get("parameters", {})
            if isinstance(args, str):
                args = json.loads(args)
            
            return tool_name, args
            
        except (json.JSONDecodeError, ValueError) as e:
            print(f"DEBUG - Parsing error: {e}")
            print(f"DEBUG - Raw JSON string: {json_str}")
            raise ValueError(f"Error parsing function call: {e}")
    
    def execute_tool(self, tool_name: str, args: dict):
        if tool_name not in self.tools:
            raise ValueError(f"Unknown tool: '{tool_name}'")
        
        result = self.tools[tool_name](**args)
        print(f"\nTOOL EXECUTED: {tool_name}")
        print(f"TOOL RESULT: {json.dumps(result, indent=2)}\n")
        
        return result
    
    def has_tool(self, tool_name: str) -> bool:
        return tool_name in self.tools
    
    def list_tool_names(self) -> list:
        return list(self.tools.keys())
    
    def add_tool(self, tool):
        self.list_of_tools.append(tool)
        self._build_tool_mappings()
    
    def add_tools(self, tools: list):
        self.list_of_tools.extend(tools)
        self._build_tool_mappings()