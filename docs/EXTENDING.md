# Extensibility Guide

This document covers all four extension points in the Fairness Evaluation System: Tools, Agent types, LLM Backends, and Pipeline Stages. Each section includes the abstract interface, a complete implementation template, and the registration steps required to wire the new component into the runtime.


## Architecture Overview

The system is a 4-layer stack:

```
LLM Backend (BaseModelClient)
    ↓  generate(messages) → str
Agent (BaseAgent)
    ↓  run(user_message) → str
Tool (ToolManager / Tool)
    ↓  execute_tool(name, args) → dict
Pipeline Stage (BaseStageExecutor)
    ↓  __call__(stage, ctx) → dict
```

| Layer | Base class | File |
|-------|-----------|------|
| LLM Backend | `BaseModelClient` | `src/models/clients/base_client.py` |
| Agent | `BaseAgent` | `src/models/agents/base_agent.py` |
| Tool | `ToolManager` + `Tool` | `src/tools/tool_manager.py`, `src/tools/tool.py` |
| Pipeline Stage | `BaseStageExecutor` | `src/pipeline/stages/base.py` |

Configuration is declarative: `src/models/config.yml` wires models, agents, and tools; `src/pipeline/pipeline_config.yml` defines stage order and executor class names.


## 1. Adding a New Tool

### What a Tool is

A `Tool` is a named, callable function with a JSON-Schema parameter spec, exposed to agents via a `ToolManager`. The LLM receives the schema as part of its system prompt and can invoke tools by emitting structured JSON.

### Step 1 — Implement the ToolManager subclass

```python
# src/tools/my_tools.py
from tools.tool import Tool
from tools.tool_manager import ToolManager


class MyTools(ToolManager):
    def __init__(self):
        super().__init__()

        # Instantiate Tool objects and assign to instance attributes
        self.tool_compute_metric = Tool(
            name="compute_my_metric",
            function=self.compute_my_metric,
            description="Compute my custom fairness metric on the dataset.",
            parameters={
                "type": "object",
                "properties": {
                    "dataset_name": {
                        "type": "string",
                        "description": "Name of the CSV dataset (without .csv extension)"
                    },
                    "column_name": {
                        "type": "string",
                        "description": "Target column to compute the metric for"
                    },
                },
                "required": ["dataset_name", "column_name"],
            },
        )

        # Register all tools in list_of_tools — this is what ToolManager exposes
        self.list_of_tools = [
            self.tool_compute_metric,
            # add more Tool instances here
        ]
        # Must be called after populating list_of_tools
        self._build_tool_mappings()

    # ── Tool implementation ───────────────────────────────────────────────────
    def compute_my_metric(self, dataset_name: str, column_name: str) -> dict:
        """
        Return a dict. Agents receive the entire dict as structured context.
        Always include a "status" key ("success" | "error").
        """
        try:
            import os, pandas as pd

            data_dir = os.path.join(
                os.path.dirname(os.path.dirname(__file__)), "data"
            )
            path = os.path.join(data_dir, f"{dataset_name}.csv")
            df = pd.read_csv(path)

            if column_name not in df.columns:
                return {"status": "error", "message": f"Column '{column_name}' not found"}

            result_value = float(df[column_name].mean())

            return {
                "status": "success",
                "dataset": dataset_name,
                "column": column_name,
                "my_metric": result_value,
            }
        except Exception as exc:
            return {"status": "error", "message": str(exc)}
```

### Step 2 — Register in AgentManager

`AgentManager._get_tool_manager()` in `src/models/agent_manager.py` resolves tool names to instances. Add a branch for your new class:

```python
# src/models/agent_manager.py  (inside AgentManager._get_tool_manager)
def _get_tool_manager(self, tool_name: str):
    if tool_name not in self._tools:
        if tool_name.lower() in ["fairness", "fairness_tools"]:
            self._tools[tool_name] = FairnessTools()
        elif tool_name.lower() in ["bias_mitigation", "bias_mitigation_tools"]:
            self._tools[tool_name] = BiasMitigationTools()
        elif tool_name.lower() in ["my_tools", "mytools"]:          # ← add this
            from tools.my_tools import MyTools
            self._tools[tool_name] = MyTools()
        else:
            raise ValueError(f"Unknown tool manager: {tool_name}")
    return self._tools[tool_name]
```

### Step 3 — Declare in config.yml

```yaml
tools:
  fairness:
    class: "FairnessTools"
    description: "..."
  bias_mitigation:
    class: "BiasMitigationTools"
    description: "..."
  my_tools:                              # key used in agents.*.tools list
    class: "MyTools"
    description: "My custom metric tools"

agents:
  my_analyst:
    type: FunctionCallerAgent
    tools: [my_tools]
    stages: [my_analysis_stage]
    reflect_on_tool_use: true
```
The `class:` value is informational only (not auto-loaded); actual loading happens in `_get_tool_manager()`.


## 2. Adding a New Agent Type

### Abstract interface

```python
# src/models/agents/base_agent.py
class BaseAgent(ABC):
    def __init__(self, model_client: BaseModelClient, model_name: str = None): ...

    @abstractmethod
    def run(self, user_message: str) -> str: ...

    @abstractmethod
    def get_system_prompt(self) -> str: ...

    def ask_model(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        max_retries: int = 3,
    ) -> str: ...
```

`ask_model` handles retries, exponential back-off, `<think>` tag stripping, and reasoning-leakage detection. Always use it instead of calling `self.model_client.generate()` directly.

### Step 1 — Implement the agent

```python
# src/models/agents/my_agent.py
from typing import List, Dict
from models.agents.base_agent import BaseAgent
from models.clients.base_client import BaseModelClient
from tools.tool_manager import ToolManager


class MyAgent(BaseAgent):
    """
    Example agent that runs a tool and then writes a structured analysis.
    """

    def __init__(
        self,
        model_client: BaseModelClient,
        tool_manager: ToolManager = None,
        model_name: str = None,
    ):
        super().__init__(model_client=model_client, model_name=model_name)
        self.tool_manager = tool_manager

    def get_system_prompt(self) -> str:
        return (
            "You are a fairness analyst. Given the tool output, write a concise "
            "Markdown section describing the findings and their implications. "
            "Be precise and cite specific numbers."
        )

    def run(self, user_message: str) -> str:
        messages: List[Dict[str, str]] = [
            {"role": "system", "content": self.get_system_prompt()},
            {"role": "user",   "content": user_message},
        ]
        # ask_model raises APIError on unrecoverable failures
        return self.ask_model(messages, temperature=0.2, max_tokens=2048)
```

### Step 2 — Register the type string in AgentManager

`_get_agent_class()` in `src/models/agent_manager.py` maps type strings to classes via substring matching:

```python
# src/models/agent_manager.py  (inside AgentManager._get_agent_class)
def _get_agent_class(self, agent_type: str):
    type_lower = agent_type.lower()

    if "function" in type_lower or "caller" in type_lower:
        return FunctionCallerAgent
    elif "analyst" in type_lower or "data" in type_lower:
        return DataAnalystAgent
    elif "conversation" in type_lower or "assistant" in type_lower:
        return ConversationalAgent
    elif "myagent" in type_lower or "my_agent" in type_lower:    # ← add this
        from models.agents.my_agent import MyAgent
        return MyAgent
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")
```

### Step 3 — Declare in config.yml

```yaml
agents:
  my_custom_agent:
    type: MyAgent                # matched by _get_agent_class substring logic
    tools: [my_tools]
    stages: [my_analysis_stage]
    reflect_on_tool_use: false
    model: gemini-flash          # optional per-agent model override
```

If your agent constructor accepts kwargs beyond `model_client` and `tool_manager` (e.g. `reflect_on_tool_use`), `AgentManager.get_agent()` passes them only when the signature declares them — it does an `inspect.signature` check before injection.


## 3. Adding a New LLM Backend

### Abstract interface

```python
# src/models/clients/base_client.py
@dataclass
class ModelInfo:
    name: str
    provider: str
    supports_vision: bool = False
    supports_function_calling: bool = False
    supports_json_output: bool = False
    supports_structured_output: bool = False
    supports_streaming: bool = False
    max_tokens: int = 4096
    context_window: int = 8192


class BaseModelClient(ABC):
    @abstractmethod
    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str: ...

    @abstractmethod
    def get_model_info(self) -> ModelInfo: ...
```

`generate` must return a plain string (the model's text response). Raise a standard `Exception` with the HTTP status code in the message on API errors — `BaseAgent.ask_model` parses the status code for retry decisions.

### Step 1 — Implement the client

```python
# src/models/clients/my_provider_client.py
import requests
from typing import List, Dict
from models.clients.base_client import BaseModelClient, ModelInfo


class MyProviderClient(BaseModelClient):
    def __init__(
        self,
        model: str = "my-model-v1",
        api_key: str = None,
        base_url: str = "https://api.myprovider.com/v1",
        **kwargs,
    ):
        super().__init__()
        self.model = model
        self.base_url = base_url.rstrip("/")
        import os
        self.api_key = api_key or os.environ.get("MY_PROVIDER_API_KEY", "")

    def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.2,
        max_tokens: int = 4096,
        **kwargs,
    ) -> str:
        self.validate_messages(messages)   # raises ValueError on malformed input

        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            f"{self.base_url}/chat/completions",
            json=payload,
            headers=headers,
            timeout=120,
        )

        if response.status_code != 200:
            # Include status code in message — BaseAgent parses it for retry logic
            raise Exception(
                f"MyProvider API error {response.status_code}: {response.text}"
            )

        data = response.json()
        return data["choices"][0]["message"]["content"]

    def get_model_info(self) -> ModelInfo:
        return ModelInfo(
            name=self.model,
            provider="myprovider",
            supports_function_calling=False,
            supports_vision=False,
            max_tokens=4096,
            context_window=16384,
        )
```

### Step 2 — Register in ClientFactory

`ClientFactory._providers` in `src/models/clients/client_factory.py` is a class-level dict populated lazily in `_ensure_providers_loaded()`:

```python
# src/models/clients/client_factory.py
from models.clients.my_provider_client import MyProviderClient   # ← import

class ClientFactory:
    @classmethod
    def _ensure_providers_loaded(cls):
        if cls._providers is None:
            cls._providers = {
                "openrouter": OpenRouterClient,
                "gemini": GeminiClient,
                "google": GeminiClient,
                "ollama": OllamaClient,
                "myprovider": MyProviderClient,   # ← register here
            }
```

### Step 3 — Declare in config.yml

```yaml
models:
  my-model:
    provider: myprovider               # must match the key in ClientFactory._providers
    model: "my-model-v1"
    base_url: "https://api.myprovider.com/v1"
    model_info:
      function_calling: false
      vision: false

default_model: "my-model"
```

Any extra keys in the model config block (e.g. `base_url`) are passed as `**kwargs` to your client constructor via `ClientFactory.from_config()`.


## 4. Adding a New Pipeline Stage

### Abstract interface

```python
# src/pipeline/stages/base.py
class BaseStageExecutor(ABC):
    @abstractmethod
    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        """Run the stage and return a result dict stored in stage_data.json."""
        pass
```

`stage` is a `StageDefinition` dataclass with attributes:

| Attribute | Type | Description |
|-----------|------|-------------|
| `stage.key` | str | Unique stage identifier (e.g. `"4_imbalance"`) |
| `stage.name` | str | Human-readable label |
| `stage.agent` | `BaseAgent` | Agent instance bound to this stage |
| `stage.user_context` | str \| None | Optional user instruction injected at runtime |
| `stage.executor` | `BaseStageExecutor` | This executor instance |

`ctx` is the shared mutable context dict that every stage reads from and writes to. Common keys populated by earlier stages:

| Key | Set by | Type |
|-----|--------|------|
| `ctx["dataset_name"]` | `LoadingStage` | str |
| `ctx["target_column"]` | `ObjectiveStage` | str \| None |
| `ctx["sensitive_columns"]` | `SensitiveDetectionStage` | list[str] |
| `ctx["report_dir"]` | Pipeline init | str |
| `ctx["mitigation_config"]` | Pipeline init | dict \| None |
| `ctx["ml_config"]` | Pipeline init | dict \| None |

The dict your executor returns is stored verbatim in `stage_data.json` under the stage key.

### Step 1 — Implement the executor

```python
# src/pipeline/stages/my_stage.py
from __future__ import annotations

from typing import Any, Dict
from pipeline.stages.base import BaseStageExecutor


class MyCustomStage(BaseStageExecutor):
    """
    Example stage: computes a custom metric and asks the agent to interpret it.
    """

    def __call__(self, stage, ctx: Dict[str, Any]) -> Dict[str, Any]:
        dataset_name = ctx.get("dataset_name", "")
        target_column = ctx.get("target_column", "")
        report_dir = ctx.get("report_dir", "")

        # 1. Call a tool on the agent's tool_manager (if available)
        tool_result = {}
        if hasattr(stage.agent, "tool_manager") and stage.agent.tool_manager:
            try:
                tool_result = stage.agent.tool_manager.execute_tool(
                    "compute_my_metric",
                    {"dataset_name": dataset_name, "column_name": target_column},
                )
            except Exception as exc:
                tool_result = {"status": "error", "message": str(exc)}

        # 2. Build a prompt and ask the agent to analyse the result
        prompt = (
            f"## Custom Metric Analysis\n\n"
            f"Dataset: {dataset_name}\n"
            f"Target: {target_column}\n\n"
            f"Tool result:\n{tool_result}\n\n"
            "Provide a Markdown analysis of the result above."
        )

        # _tool_then_analyze appends stage.user_context and calls stage.agent.run()
        return self._tool_then_analyze(
            tool_name="compute_my_metric",
            tool_result=tool_result,
            prompt=prompt,
            stage=stage,
            # Optional extra fields stored in the result dict:
            my_custom_field="stored verbatim in stage_data.json",
        )
```

For stages that need to write files (e.g. images), write to `ctx["report_dir"]` and include the paths in the returned dict.

### Step 2 — Register in the executor registry

`_EXECUTOR_REGISTRY` in `src/pipeline/config.py` maps YAML class name strings to executor classes:

```python
# src/pipeline/config.py
from pipeline.stages.my_stage import MyCustomStage    # ← import

_EXECUTOR_REGISTRY: Dict[str, Type[BaseStageExecutor]] = {
    "LoadingStage": LoadingStage,
    "ObjectiveStage": ObjectiveStage,
    "QualityStage": QualityStage,
    "SensitiveDetectionStage": SensitiveDetectionStage,
    "DiscretizationStage": DiscretizationStage,
    "ImbalanceStage": ImbalanceStage,
    "TargetFairnessStage": TargetFairnessStage,
    "RecommendationsStage": RecommendationsStage,
    "MitigationStage": MitigationStage,
    "MyCustomStage": MyCustomStage,    # ← register here
}
```

Also add the import at the top of `src/pipeline/stages/__init__.py` if you want it re-exported:

```python
# src/pipeline/stages/__init__.py
from pipeline.stages.my_stage import MyCustomStage
```

### Step 3 — Add to pipeline_config.yml

`src/pipeline/pipeline_config.yml` defines stage order, executor binding, and optional behaviour flags:

```yaml
stages:
  - key: "0_loading"
    name: "Dataset Loading"
    executor: LoadingStage
    agent: file_parser_agent
    description: "Load and validate the dataset file."

  # ... existing stages ...

  - key: "7_my_custom"
    name: "My Custom Analysis"
    executor: MyCustomStage          # must match a key in _EXECUTOR_REGISTRY
    agent: my_custom_agent           # must match an attribute on DatasetEvaluationPipeline
    description: "Runs my custom metric and generates an analysis."
    optional: true                   # stage is skipped if prerequisites are missing
    requires_target: true            # only included when target_column is set
    requires_confirmation: true      # GUI pauses for user confirmation before running
```

The `agent:` value must be an **attribute name on `DatasetEvaluationPipeline`** (`src/pipeline/pipeline.py`). If you are adding a new agent to an existing attribute slot, no pipeline changes are needed. If you need a new dedicated agent attribute, add it to `DatasetEvaluationPipeline.__init__` and wire it via `AgentManager.get_agent()`.

### Step 4 — Expose the agent attribute on the pipeline (if needed)

```python
# src/pipeline/pipeline.py  (inside DatasetEvaluationPipeline.__init__)
self.my_custom_agent = self.agent_manager.get_agent("my_custom_agent")
```

And declare `my_custom_agent` in `config.yml`:

```yaml
agents:
  my_custom_agent:
    type: MyAgent
    tools: [my_tools]
    stages: [my_analysis_stage]
```


## Summary: Registration Checklist

| What you're adding | Files to modify |
|--------------------|----------------|
| New Tool | `src/tools/my_tools.py` (new file) · `src/models/agent_manager.py` (`_get_tool_manager`) · `config.yml` (`tools:` + `agents.*.tools`) |
| New Agent type | `src/models/agents/my_agent.py` (new file) · `src/models/agent_manager.py` (`_get_agent_class`) · `config.yml` (`agents.*.type`) |
| New LLM Backend | `src/models/clients/my_provider_client.py` (new file) · `src/models/clients/client_factory.py` (`_providers` dict) · `config.yml` (`models:`) |
| New Pipeline Stage | `src/pipeline/stages/my_stage.py` (new file) · `src/pipeline/stages/__init__.py` · `src/pipeline/config.py` (`_EXECUTOR_REGISTRY`) · `src/pipeline/pipeline_config.yml` · `src/pipeline/pipeline.py` (if new agent attribute needed) |
