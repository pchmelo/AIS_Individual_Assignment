from agents.base_agent import BaseAgent
from agents.model_client import BaseModelClient

class ConversationalAgent(BaseAgent):    
    def __init__(self, model_client: BaseModelClient = None, model_name: str = None):
        super().__init__(model_client=model_client, model_name=model_name)
    
    def get_system_prompt(self) -> str:
        return """You are a friendly and helpful conversational assistant.
                  You provide clear, concise, and accurate responses to user questions.
                  You are polite, professional, and engaging.
               """
    
    def run(self, user_message: str, max_tokens: int = 1024) -> str:
        messages = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user", "content": user_message}
        ]
        
        response = self.ask_model(messages, max_tokens=max_tokens)
        return response