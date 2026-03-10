import os
import yaml
from typing import Dict, Any, List, Optional
from models.clients.base_client import BaseModelClient
from models.clients.client_factory import ClientFactory

from tools.fairness_tools import FairnessTools
from tools.bias_mitigation_tools import BiasMitigationTools

from models.agents.conversational_agent import ConversationalAgent
from models.agents.data_analyst_agent import DataAnalystAgent
from models.agents.function_caller_agent import FunctionCallerAgent


class AgentConfig:    
    def __init__(
        self,
        name: str,
        agent_type: str,
        tools: List[str] = None,
        stages: List[str] = None,
        model: str = None,
        reflect_on_tool_use: bool = True,
        **kwargs
    ):
        self.name = name
        self.agent_type = agent_type
        self.tools = tools or []
        self.stages = stages or []
        self.model = model
        self.reflect_on_tool_use = reflect_on_tool_use
        self.extra_config = kwargs
    
    def __repr__(self):
        return (
            f"AgentConfig(name='{self.name}', type='{self.agent_type}', "
            f"model='{self.model}', stages={self.stages})"
        )


class WorkflowPhase:    
    def __init__(self, name: str, agents: List[str] = None, description: str = ""):
        self.name = name
        self.agents = agents or []
        self.description = description
    
    def __repr__(self):
        return f"WorkflowPhase(name='{self.name}', agents={self.agents})"


class AgentManager:
    """
    Manages agent configuration, instantiation, and pipeline stage associations.    
    """
    
    def __init__(self, config: Dict[str, Any] = None, api_key: str = None):
        self.config = config or {}
        self.api_key = api_key
        
        # Cached instances
        self._agents: Dict[str, Any] = {}
        self._tools: Dict[str, Any] = {}

        # Dict of model_name -> BaseModelClient 
        self._clients: Dict[str, BaseModelClient] = {}
        
        # Configuration storage
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._stages: Dict[str, WorkflowPhase] = {}
        
        # Load config if provided
        if config:
            self._parse_config(config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str, api_key: str = None) -> "AgentManager":
        """
        Create AgentManager from a YAML configuration file.
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(config, api_key=api_key)
    
    def _parse_config(self, config: Dict[str, Any]):
        self.config = config
        
        agents_config = config.get("agents", {})
        _AGENT_KNOWN_KEYS = {
            "type", "tools", "stages",
            "model", "reflect_on_tool_use",
        }
        
        for name, agent_cfg in agents_config.items():
            if isinstance(agent_cfg, dict):
                self._agent_configs[name] = AgentConfig(
                    name=name,
                    agent_type=agent_cfg.get("type", "ConversationalAgent"),
                    tools=agent_cfg.get("tools", []),
                    stages=agent_cfg.get("stages", []),
                    model=agent_cfg.get("model"),
                    reflect_on_tool_use=agent_cfg.get("reflect_on_tool_use", True),
                    **{k: v for k, v in agent_cfg.items()
                       if k not in _AGENT_KNOWN_KEYS}
                )
        
        # Parse pipeline configuration
        pipeline_config = config.get("pipeline", {})
        stages_config = pipeline_config.get("stages", [])
        
        for i, stage_cfg in enumerate(stages_config):
            if isinstance(stage_cfg, dict):
                name = stage_cfg.get("name", f"stage_{i}")
                self._stages[name] = WorkflowPhase(
                    name=name,
                    agents=stage_cfg.get("agents", []),
                    description=stage_cfg.get("description", "")
                )
            elif isinstance(stage_cfg, str):
                self._stages[stage_cfg] = WorkflowPhase(name=stage_cfg)
    
    # ------------------------------------------------------------------
    # Client / model helpers
    # ------------------------------------------------------------------
    
    @property
    def _default_model_name(self) -> Optional[str]:
        """Get the default model name from config."""
        return self.config.get("default_model")
    
    def get_client(self, model_name: str = None) -> Optional[BaseModelClient]:
        """
        Get (or lazily create) a model client.
        """
        key = model_name or self._default_model_name
        if key and key not in self._clients:
            if self.config.get("models"):
                # Inject api_key into config if provided
                config = self.config
                if self.api_key:
                    config = config.copy()
                    models = config.get("models", {}).copy()
                    if key in models:
                        models[key] = models[key].copy()
                        models[key]["api_key"] = self.api_key
                    config["models"] = models
                self._clients[key] = ClientFactory.from_yaml_config(config, key)
        return self._clients.get(key)
    
    def _get_tool_manager(self, tool_name: str):
        """Get or create a tool manager instance."""
        if tool_name not in self._tools:
            if tool_name.lower() in ["fairness", "fairness_tools"]:
                self._tools[tool_name] = FairnessTools()
            elif tool_name.lower() in ["bias_mitigation", "bias_mitigation_tools"]:
                self._tools[tool_name] = BiasMitigationTools()
            else:
                raise ValueError(f"Unknown tool manager: {tool_name}")
        
        return self._tools[tool_name]
    
    def _get_agent_class(self, agent_type: str):
        """Get agent class by type name."""
        type_lower = agent_type.lower()
        
        if "function" in type_lower or "caller" in type_lower:
            return FunctionCallerAgent
        elif "analyst" in type_lower or "data" in type_lower:
            return DataAnalystAgent
        elif "conversation" in type_lower or "assistant" in type_lower:
            return ConversationalAgent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def get_agent(self, agent_name: str):
        """
        Get or create an agent by name.
        """
        if agent_name not in self._agents:
            if agent_name not in self._agent_configs:
                raise ValueError(f"Agent not found in config: {agent_name}")
            
            config = self._agent_configs[agent_name]
            agent_class = self._get_agent_class(config.agent_type)
            
            client = self.get_client(config.model)
            
            kwargs = {
                "model_client": client
            }
            
            # Add tools if agent supports them
            if config.tools:
                primary_tool = self._get_tool_manager(config.tools[0])
                for additional_tool in config.tools[1:]:
                    tool_manager = self._get_tool_manager(additional_tool)
                    primary_tool.add_tools(tool_manager.list_of_tools)
                kwargs["tool_manager"] = primary_tool
            
            # Add reflection setting for FunctionCallerAgent
            if hasattr(agent_class, '__init__'):
                import inspect
                sig = inspect.signature(agent_class.__init__)
                if 'reflect_on_tool_use' in sig.parameters:
                    kwargs["reflect_on_tool_use"] = config.reflect_on_tool_use
            
            # Add any extra config
            kwargs.update(config.extra_config)
            
            # Create agent
            try:
                self._agents[agent_name] = agent_class(**kwargs)
            except TypeError as e:
                # Remove unsupported kwargs and try again
                kwargs = {"model_client": client}
                if config.tools:
                    kwargs["tool_manager"] = self._get_tool_manager(config.tools[0])
                self._agents[agent_name] = agent_class(**kwargs)
        
        return self._agents[agent_name]
    
    def get_agents_for_stage(self, stage_name: str) -> List:
        """
        Get all agents associated with a pipeline stage.
        """
        agents = []
        
        # Check stage configuration
        if stage_name in self._stages:
            stage = self._stages[stage_name]
            for agent_name in stage.agents:
                agents.append(self.get_agent(agent_name))
        
        # Also check agents that have this stage in their config
        for name, config in self._agent_configs.items():
            if stage_name in config.stages and name not in [a for a in (self._stages.get(stage_name) or WorkflowPhase("")).agents]:
                agents.append(self.get_agent(name))
        
        return agents
    
    def get_primary_agent_for_stage(self, stage_name: str):
        """
        Get the primary (first) agent for a pipeline stage.
        """
        agents = self.get_agents_for_stage(stage_name)
        return agents[0] if agents else None
    
    def __repr__(self):
        return (
            f"AgentManager(agents={len(self._agent_configs)}, "
            f"stages={len(self._stages)}, "
            f"models_loaded={len(self._clients)})"
        )
