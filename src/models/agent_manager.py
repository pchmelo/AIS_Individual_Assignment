import os
import yaml
from typing import Dict, Any, List, Optional, Type
from models.clients.base_client import BaseModelClient
from models.clients.client_factory import ClientFactory


_AGENT_REGISTRY: Dict[str, Type] = {}

_TOOL_REGISTRY: Dict[str, Type] = {}


def register_agent(name: str):
    def decorator(cls):
        _AGENT_REGISTRY[name.lower()] = cls
        return cls
    return decorator


def register_tool_manager(name: str):
    def decorator(cls):
        _TOOL_REGISTRY[name.lower()] = cls
        return cls
    return decorator


class AgentConfig:    
    def __init__(
        self,
        name: str,
        agent_type: str,
        tools: List[str] = None,
        stages: List[str] = None,
        model: str = None,
        client_override: str = None,
        reflect_on_tool_use: bool = True,
        **kwargs
    ):
        self.name = name
        self.agent_type = agent_type
        self.tools = tools or []
        self.stages = stages or []
        self.model = model or client_override
        self.reflect_on_tool_use = reflect_on_tool_use
        self.extra_config = kwargs
    
    def __repr__(self):
        return (
            f"AgentConfig(name='{self.name}', type='{self.agent_type}', "
            f"model='{self.model}', stages={self.stages})"
        )


class PipelineStage:    
    def __init__(self, name: str, agents: List[str] = None, description: str = ""):
        self.name = name
        self.agents = agents or []
        self.description = description
    
    def __repr__(self):
        return f"PipelineStage(name='{self.name}', agents={self.agents})"


class AgentManager:
    """
    Manages agent configuration, instantiation, and pipeline stage associations.
    
    Features:
    - Load configuration from YAML files
    - Create and cache agent instances
    - Associate agents with tools
    - Map agents to pipeline stages
    - Support for multiple model clients
    
    Example:
        manager = AgentManager.from_yaml("config.yml")
        
        # Get agent for a specific stage
        agent = manager.get_agent_for_stage("parsing")
        
        # Run agent
        result = agent.run("Analyze this dataset")
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize AgentManager.
        
        Args:
            config: Configuration dictionary (optional, can load later)
        """
        self.config = config or {}
        
        # Cached instances
        self._agents: Dict[str, Any] = {}
        self._tools: Dict[str, Any] = {}
        # Dict of model_name -> BaseModelClient (lazy-created)
        self._clients: Dict[str, BaseModelClient] = {}
        
        # Configuration storage
        self._agent_configs: Dict[str, AgentConfig] = {}
        self._stages: Dict[str, PipelineStage] = {}
        self._stage_order: List[str] = []
        
        # Load config if provided
        if config:
            self._parse_config(config)
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "AgentManager":
        """
        Create AgentManager from a YAML configuration file.
        
        Args:
            yaml_path: Path to YAML configuration file
        
        Returns:
            Configured AgentManager instance
        """
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return cls(config)
    
    def _parse_config(self, config: Dict[str, Any]):
        """Parse configuration dictionary."""
        self.config = config
        
        # Parse tools configuration
        tools_config = config.get("tools", {})
        for tool_name, tool_config in tools_config.items():
            # Tool configs are stored for lazy instantiation
            pass
        
        # Parse agents configuration
        agents_config = config.get("agents", {})
        _AGENT_KNOWN_KEYS = {
            "type", "agent_type", "tools", "stages",
            "model", "client", "reflect_on_tool_use",
        }
        
        # Handle list format (old style)
        if isinstance(agents_config, list):
            for i, agent_cfg in enumerate(agents_config):
                name = agent_cfg.get("name", f"agent_{i}")
                self._agent_configs[name] = AgentConfig(
                    name=name,
                    agent_type=agent_cfg.get("agent_type", "ConversationalAgent"),
                    tools=agent_cfg.get("tools", []),
                    stages=agent_cfg.get("stages", []),
                    model=agent_cfg.get("model"),
                    client_override=agent_cfg.get("client"),
                    reflect_on_tool_use=agent_cfg.get("reflect_on_tool_use", True)
                )
        # Handle dict format (new style)
        else:
            for name, agent_cfg in agents_config.items():
                if isinstance(agent_cfg, dict):
                    self._agent_configs[name] = AgentConfig(
                        name=name,
                        agent_type=agent_cfg.get("type", agent_cfg.get("agent_type", "ConversationalAgent")),
                        tools=agent_cfg.get("tools", []),
                        stages=agent_cfg.get("stages", []),
                        model=agent_cfg.get("model"),
                        client_override=agent_cfg.get("client"),
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
                self._stages[name] = PipelineStage(
                    name=name,
                    agents=stage_cfg.get("agents", []),
                    description=stage_cfg.get("description", "")
                )
                self._stage_order.append(name)
            elif isinstance(stage_cfg, str):
                self._stages[stage_cfg] = PipelineStage(name=stage_cfg)
                self._stage_order.append(stage_cfg)
    
    # ------------------------------------------------------------------
    # Client / model helpers
    # ------------------------------------------------------------------

    def set_client(self, client: BaseModelClient, model_name: str = None):
        """
        Set a model client, optionally for a specific model name.
        
        Args:
            client: BaseModelClient instance
            model_name: Model name to associate with. If None, sets as the
                        default model client.
        """
        key = model_name or self._default_model_name
        self._clients[key] = client
    
    @property
    def _default_model_name(self) -> Optional[str]:
        """Get the default model name from config."""
        # New format
        name = self.config.get("default_model")
        if name:
            return name
        # Legacy format
        clients_cfg = self.config.get("clients", {})
        return clients_cfg.get("default")
    
    def get_client(self, model_name: str = None) -> Optional[BaseModelClient]:
        """
        Get (or lazily create) a model client.
        
        Args:
            model_name: Name of the model from config. If None, returns
                        the default model client.
        
        Returns:
            BaseModelClient instance
        """
        key = model_name or self._default_model_name
        if key and key not in self._clients:
            # Lazy create from config
            has_models = self.config.get("models") or self.config.get("clients")
            if has_models:
                self._clients[key] = ClientFactory.from_yaml_config(self.config, key)
        return self._clients.get(key)
    
    def list_available_models(self) -> Dict[str, Any]:
        """
        List all model definitions from the config.
        
        Returns:
            Dict of model_name -> config dict
        """
        return ClientFactory.list_models_from_config(self.config)
    
    def _get_tool_manager(self, tool_name: str):
        """Get or create a tool manager instance."""
        if tool_name not in self._tools:
            # Lazy import tool managers
            if tool_name.lower() in ["fairness", "fairness_tools"]:
                from tools.fairness_tools import FairnessTools
                self._tools[tool_name] = FairnessTools()
            elif tool_name.lower() in ["bias_mitigation", "bias_mitigation_tools"]:
                from tools.bias_mitigation_tools import BiasMitigationTools
                self._tools[tool_name] = BiasMitigationTools()
            elif tool_name in _TOOL_REGISTRY:
                self._tools[tool_name] = _TOOL_REGISTRY[tool_name]()
            else:
                raise ValueError(f"Unknown tool manager: {tool_name}")
        
        return self._tools[tool_name]
    
    def _get_agent_class(self, agent_type: str):
        """Get agent class by type name."""
        # Lazy import agent classes
        type_lower = agent_type.lower()
        
        if type_lower in _AGENT_REGISTRY:
            return _AGENT_REGISTRY[type_lower]
        
        if "function" in type_lower or "caller" in type_lower:
            from models.agents.function_caller_agent import FunctionCallerAgent
            return FunctionCallerAgent
        elif "analyst" in type_lower or "data" in type_lower:
            from models.agents.data_analyst_agent import DataAnalystAgent
            return DataAnalystAgent
        elif "conversation" in type_lower or "assistant" in type_lower:
            from models.agents.conversational_agent import ConversationalAgent
            return ConversationalAgent
        else:
            raise ValueError(f"Unknown agent type: {agent_type}")
    
    def get_agent(self, agent_name: str):
        """
        Get or create an agent by name.
        
        Args:
            agent_name: Name of the agent from config
        
        Returns:
            Agent instance
        """
        if agent_name not in self._agents:
            if agent_name not in self._agent_configs:
                raise ValueError(f"Agent not found in config: {agent_name}")
            
            config = self._agent_configs[agent_name]
            agent_class = self._get_agent_class(config.agent_type)
            
            # Get client – use agent-specific model if set, otherwise default
            client = self.get_client(config.model)
            
            # Build agent kwargs
            kwargs = {
                "model_client": client
            }
            
            # Add tools if agent supports them
            if config.tools:
                # Combine multiple tool managers
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
        
        Args:
            stage_name: Name of the pipeline stage
        
        Returns:
            List of agent instances
        """
        agents = []
        
        # Check stage configuration
        if stage_name in self._stages:
            stage = self._stages[stage_name]
            for agent_name in stage.agents:
                agents.append(self.get_agent(agent_name))
        
        # Also check agents that have this stage in their config
        for name, config in self._agent_configs.items():
            if stage_name in config.stages and name not in [a for a in (self._stages.get(stage_name) or PipelineStage("")).agents]:
                agents.append(self.get_agent(name))
        
        return agents
    
    def get_primary_agent_for_stage(self, stage_name: str):
        """
        Get the primary (first) agent for a pipeline stage.
        
        Args:
            stage_name: Name of the pipeline stage
        
        Returns:
            Agent instance or None
        """
        agents = self.get_agents_for_stage(stage_name)
        return agents[0] if agents else None
    
    def get_stage_order(self) -> List[str]:
        """Get the ordered list of pipeline stages."""
        return self._stage_order.copy()
    
    def list_agents(self) -> List[str]:
        """Get list of all configured agent names."""
        return list(self._agent_configs.keys())
    
    def list_stages(self) -> List[str]:
        """Get list of all configured pipeline stages."""
        return list(self._stages.keys())
    
    def get_agent_config(self, agent_name: str) -> Optional[AgentConfig]:
        """Get configuration for a specific agent."""
        return self._agent_configs.get(agent_name)
    
    def add_agent_config(
        self,
        name: str,
        agent_type: str,
        tools: List[str] = None,
        stages: List[str] = None,
        **kwargs
    ):
        """
        Dynamically add an agent configuration.
        
        Args:
            name: Unique agent name
            agent_type: Type of agent (FunctionCallerAgent, DataAnalystAgent, etc.)
            tools: List of tool manager names
            stages: List of pipeline stages this agent participates in
            **kwargs: Additional agent configuration
        """
        self._agent_configs[name] = AgentConfig(
            name=name,
            agent_type=agent_type,
            tools=tools,
            stages=stages,
            **kwargs
        )
    
    def add_stage(self, name: str, agents: List[str] = None, position: int = None):
        self._stages[name] = PipelineStage(name=name, agents=agents or [])
        
        if position is not None:
            self._stage_order.insert(position, name)
        else:
            self._stage_order.append(name)
    
    def clear_cache(self):
        self._agents.clear()
        self._tools.clear()
        self._clients.clear()
    
    def __repr__(self):
        return (
            f"AgentManager(agents={len(self._agent_configs)}, "
            f"stages={len(self._stages)}, "
            f"models_loaded={len(self._clients)})"
        )
        

def create_agent_manager(config_path: str = None, config: Dict = None) -> AgentManager:
    if config_path:
        return AgentManager.from_yaml(config_path)
    elif config:
        return AgentManager(config)
    else:
        return AgentManager()
