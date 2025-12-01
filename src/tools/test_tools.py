import pandas as pd
import json
import urllib.request, urllib.parse
from tools.tool import Tool
from tools.tool_manager import ToolManager

class TestTools(ToolManager):
    def __init__(self):
        super().__init__()
        
        self.tool_current_temp = Tool(
            name="current_temp",
            function=self.current_temp,
            description="Get the current temperature of a city",
            parameters={
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Name of the city"}
                },
                "required": ["city"]
            }
        )

        self.tool_read_csv = Tool(
            name="read_csv",
            function=self.read_csv,
            description="Read a CSV file and return its columns and a preview of its content",
            parameters={
                "type": "object",
                "properties": {
                    "path": {"type": "string", "description": "Path to the CSV file"}
                },
                "required": ["path"]
            }
        )
        
        self.list_of_tools = [self.tool_current_temp, self.tool_read_csv]
        self._build_tool_mappings()

    @staticmethod
    def current_temp(city: str) -> dict:
        q = urllib.parse.quote_plus(city.strip())
        geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={q}&count=1"

        with urllib.request.urlopen(geo_url, timeout=8) as r:
            g = json.load(r)

        if not g.get("results"):
            raise ValueError(f"City '{city}' not found.")

        loc = g["results"][0]
        lat, lon = loc["latitude"], loc["longitude"]

        wx_url = (f"https://api.open-meteo.com/v1/forecast"
                f"?latitude={lat}&longitude={lon}"
                f"&current=temperature_2m&timezone=auto")

        with urllib.request.urlopen(wx_url, timeout=8) as r:
            w = json.load(r)

        temp = float(w["current"]["temperature_2m"])
        unit = w.get("current_units", {}).get("temperature_2m", "°C")

        return {
            "temperature": temp,
            "unit": unit
        }
    
    @staticmethod
    def read_csv(path: str):
        df = pd.read_csv(path)
        return {
            "columns": list(df.columns),
            "rows": len(df),
            "preview": df.head(5).to_dict(orient="records")
        }
    
test_tools = TestTools()