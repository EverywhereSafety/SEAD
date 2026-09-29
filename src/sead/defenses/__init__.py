"""SAGE runtime interfaces."""

from importlib import import_module

# Load runtime components only when requested.
_EXPORTS = {
    "build_tool_defender": ".defender",
    "SAGEDefenseError": ".sage",
    "SAGEDefenseTool": ".sage",
    "SAGEDefender": ".sage",
    "BatchLLMClient": ".client",
    "Action": ".base",
    "BaseToolDefender": ".base",
    "DEFAULT_BLOCKED_OBSERVATION": ".online",
    "ONLINE_DEFENDER_TYPES": ".online",
    "ONLINE_TOOL_DEFENDER_TYPES": ".online",
    "OnlineToolDefenseError": ".online",
    "OnlineToolDefenseGate": ".online",
    "OpenAICompatibleBatchLLMClient": ".online",
    "build_online_tool_defender": ".online",
    "configured_openhands_tool_defense": ".online",
    "guard_openhands_runtime": ".online",
    "validate_online_defense_config": ".online",
    "validate_online_tool_defense_config": ".online",
}

__all__ = [
    "BatchLLMClient",
    "build_tool_defender",
    "DEFAULT_BLOCKED_OBSERVATION",
    "ONLINE_DEFENDER_TYPES",
    "ONLINE_TOOL_DEFENDER_TYPES",
    "Action",
    "SAGEDefenseError",
    "SAGEDefenseTool",
    "SAGEDefender",
    "BaseToolDefender",
    "OnlineToolDefenseError",
    "OnlineToolDefenseGate",
    "OpenAICompatibleBatchLLMClient",
    "build_online_tool_defender",
    "configured_openhands_tool_defense",
    "guard_openhands_runtime",
    "validate_online_defense_config",
    "validate_online_tool_defense_config",
]


def __getattr__(name: str):
    try:
        module = _EXPORTS[name]
    except KeyError:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}") from None
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
