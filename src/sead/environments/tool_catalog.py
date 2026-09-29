"""Actual runtime tool schemas, cached only with a matching environment identity."""
from dataclasses import dataclass
import json
from pathlib import Path

from jsonschema import Draft202012Validator, ValidationError
from sead.hashing import sha256_json


CONTROL_FLOW_TOOLS = frozenset({"finish", "think", "end_task", "condensation_request"})


@dataclass(frozen=True)
class ToolCatalog:
    tools: tuple[dict, ...]
    identity: dict

    def __post_init__(self):
        names = []
        for tool in self.tools:
            function = tool["function"]
            names.append(function["name"])
            Draft202012Validator.check_schema(function["parameters"])
        if not names or len(set(names)) != len(names):
            raise ValueError("tool catalog is empty or contains duplicate tool names")

    def validate_call(self, name, parameters):
        if name in CONTROL_FLOW_TOOLS:
            raise ValueError("control-flow tools cannot be chain actions")
        function = next((t["function"] for t in self.tools if t["function"]["name"] == name), None)
        if function is None or not isinstance(parameters, dict):
            raise ValueError(f"unknown tool or invalid arguments: {name}")
        try:
            Draft202012Validator(function["parameters"]).validate(parameters)
        except ValidationError as exc:
            raise ValueError(f"invalid arguments for {name}: {exc.message}") from exc

    @property
    def action_tools(self):
        """Schemas exposed to attack roles that must produce executable chain actions."""

        return tuple(tool for tool in self.tools
                     if tool["function"]["name"] not in CONTROL_FLOW_TOOLS)

    @property
    def digest(self):
        return sha256_json(self.to_dict())

    def to_dict(self):
        return {"schema_version": "sead-tool-catalog-v1", "identity": self.identity,
                "tools": list(self.tools), "mcp_native_action": "call_tool_mcp"}

    @classmethod
    def read(cls, path, identity):
        value = json.loads(Path(path).read_text())
        if value.get("schema_version") != "sead-tool-catalog-v1" or value.get("identity") != identity:
            raise ValueError("tool catalog cache does not match task/runtime/MCP identity")
        return cls(tuple(value["tools"]), value["identity"])
