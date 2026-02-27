"""
Models Module

This module contains model clients, agents, and the agent manager.
"""

from models.clients import (
    BaseModelClient,
    OpenRouterClient,
    GeminiClient,
    LocalModelClient,
    ClientFactory
)
from models.agent_manager import AgentManager, create_agent_manager
from models.agents import (
    BaseAgent,
    FunctionCallerAgent,
    DataAnalystAgent,
    ConversationalAgent
)

__all__ = [
    # Clients
    'BaseModelClient',
    'OpenRouterClient',
    'GeminiClient',
    'LocalModelClient',
    'ClientFactory',
    
    # Agents
    'BaseAgent',
    'FunctionCallerAgent',
    'DataAnalystAgent',
    'ConversationalAgent',
    
    # Manager
    'AgentManager',
    'create_agent_manager'
]
