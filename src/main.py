from agents.agent import Agent
from tools.test_tools import test_tools

if __name__ == "__main__":
    agent = Agent(tool_manager=test_tools)
    
    user_question = "What is the temperature in Porto?"
    
    response = agent.run_agent(user_question)
    print(f"Agent Response: {response}")