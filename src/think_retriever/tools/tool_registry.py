"""
tool_registry.py
────────────────
Schema-driven function registry for Hermes-style function calling.

A "function" is a triple of:
  * **name**       — unique identifier the model writes inside ``<tool_call>``
  * **callable**   — Python implementation; takes the parsed ``arguments`` dict
                     as keyword args and returns a str / JSON-serialisable obj
  * **schema**     — OpenAI-style JSON Schema describing parameters
                     (rendered into the system prompt so the model knows the
                     calling convention)

This replaces the XML-tag dispatch from the agentic-RAG era. The same registry
hosts arbitrary functions, with ``search`` (the retriever) registered as just
one of many — preserving the original RAG capability while generalising to
broader tool use.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger(__name__)


# Hermes-style call/response patterns
_TOOL_CALL_RE = re.compile(
    r"<tool_call>\s*(\{.*?\})\s*</tool_call>",
    re.DOTALL,
)
_TOOL_RESPONSE_RE = re.compile(
    r"<tool_response>.*?</tool_response>",
    re.DOTALL,
)


# ── Data classes ──────────────────────────────────────────────────────────────


@dataclass
class FunctionSpec:
    """Schema-driven function specification."""

    name: str
    description: str
    parameters: Dict[str, Any]   # JSON Schema for arguments
    callable: Callable[..., Any]  # implementation

    def to_openai_schema(self) -> Dict[str, Any]:
        """Render as an OpenAI / Anthropic tool-use schema dict."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


@dataclass
class ToolCallParse:
    """One parsed <tool_call> block extracted from model text."""

    name: str
    arguments: Dict[str, Any]
    span: tuple              # (start, end) character offsets in source text
    raw_json: str = ""


@dataclass
class ExecutionResult:
    """Outcome of executing one function call."""

    name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    error_message: Optional[str] = None

    def to_response_payload(self) -> str:
        """Serialise into the string that goes inside <tool_response>."""
        if not self.success:
            return json.dumps({"error": self.error_message})
        if isinstance(self.result, (dict, list)):
            return json.dumps(self.result, ensure_ascii=False)
        return str(self.result)


# ── Registry ──────────────────────────────────────────────────────────────────


class ToolRegistry:
    """
    Registry of schema-driven functions for the agent to call.

    Typical usage:

        registry = ToolRegistry()
        registry.register_function(
            name="search",
            description="Retrieve relevant documents from the corpus.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"}
                },
                "required": ["query"],
            },
            callable=lambda **kw: retriever.search(kw["query"]),
        )
    """

    def __init__(self, response_search_window: int = 50) -> None:
        """
        Initialize the tool registry.

        Parameters
        ----------
        response_search_window:
            Maximum number of characters after a tool call to search for
            a corresponding tool response. Defaults to 50.
        """
        self._functions: Dict[str, FunctionSpec] = {}
        self.response_search_window = response_search_window

    # ── Registration ──────────────────────────────────────────────────────────

    def register_function(
        self,
        name: str,
        description: str,
        parameters: Dict[str, Any],
        callable: Callable[..., Any],
    ) -> None:
        """Register a function with its JSON-schema description."""
        if name in self._functions:
            logger.warning("Overwriting existing function registration: %s", name)
        self._functions[name] = FunctionSpec(
            name=name,
            description=description,
            parameters=parameters,
            callable=callable,
        )

    def register_from_schema(
        self,
        schema: Dict[str, Any],
        callable: Callable[..., Any],
    ) -> None:
        """
        Convenience: register from an OpenAI-style schema dict.

        ``schema`` may be either:
          * the inner function dict ``{"name": ..., "description": ..., "parameters": ...}``
          * the wrapped form ``{"type": "function", "function": {...}}``
        """
        if schema.get("type") == "function":
            schema = schema["function"]
        self.register_function(
            name=schema["name"],
            description=schema.get("description", ""),
            parameters=schema.get("parameters", {"type": "object", "properties": {}}),
            callable=callable,
        )

    def has_function(self, name: str) -> bool:
        return name in self._functions

    def list_functions(self) -> List[str]:
        return list(self._functions.keys())

    def get_schema(self, name: str) -> Optional[Dict[str, Any]]:
        spec = self._functions.get(name)
        return spec.to_openai_schema() if spec else None

    # ── Prompt rendering ──────────────────────────────────────────────────────

    def render_schemas_for_prompt(self) -> str:
        """
        Render all registered functions into a system-prompt-ready JSON block.

        Output is one JSON object per line, each on its own line, wrapped in
        a marker that the model is trained (during SFT) to recognise.
        """
        if not self._functions:
            return ""
        lines = ["<tools>"]
        for spec in self._functions.values():
            lines.append(json.dumps(spec.to_openai_schema(), ensure_ascii=False))
        lines.append("</tools>")
        return "\n".join(lines)

    # ── Parsing ───────────────────────────────────────────────────────────────

    def detect_tool_calls(self, text: str) -> List[ToolCallParse]:
        """
        Extract Hermes-style ``<tool_call>{...}</tool_call>`` blocks.

        Calls whose JSON does not parse or whose ``name`` is unknown are
        skipped — the format reward catches those upstream.
        """
        parsed: List[ToolCallParse] = []
        for m in _TOOL_CALL_RE.finditer(text):
            payload_str = m.group(1).strip()
            try:
                payload = json.loads(payload_str)
            except (json.JSONDecodeError, ValueError):
                continue
            if not isinstance(payload, dict):
                continue
            name = payload.get("name")
            args = payload.get("arguments", payload.get("parameters", {}))
            if not isinstance(name, str) or not isinstance(args, dict):
                continue
            parsed.append(
                ToolCallParse(
                    name=name,
                    arguments=args,
                    span=m.span(),
                    raw_json=payload_str,
                )
            )
        return parsed

    def detect_unprocessed_calls(self, text: str) -> List[ToolCallParse]:
        """
        Like ``detect_tool_calls`` but skips calls that already have a
        ``<tool_response>`` block immediately following them. This is what the
        rollout loop wants — it only needs to execute calls that haven't been
        executed yet.
        """
        all_calls = self.detect_tool_calls(text)
        response_starts = [m.start() for m in _TOOL_RESPONSE_RE.finditer(text)]
        unprocessed: List[ToolCallParse] = []
        for call in all_calls:
            # The call is unprocessed if NO tool_response starts within the
            # configured window of its end (i.e. before significant other generation).
            call_end = call.span[1]
            if any(
                call_end <= s <= call_end + self.response_search_window
                for s in response_starts
            ):
                continue
            unprocessed.append(call)
        return unprocessed

    # ── Execution ─────────────────────────────────────────────────────────────

    def execute(self, call: ToolCallParse) -> ExecutionResult:
        """Dispatch one parsed call to its registered implementation."""
        spec = self._functions.get(call.name)
        if spec is None:
            return ExecutionResult(
                name=call.name,
                arguments=call.arguments,
                result=None,
                success=False,
                error_message=f"Function '{call.name}' not registered.",
            )
        try:
            result = spec.callable(**call.arguments)
            return ExecutionResult(
                name=call.name,
                arguments=call.arguments,
                result=result,
                success=True,
            )
        except TypeError as exc:
            # Most common: wrong arguments (missing required / unexpected kwarg)
            return ExecutionResult(
                name=call.name,
                arguments=call.arguments,
                result=None,
                success=False,
                error_message=f"Argument error: {exc}",
            )
        except Exception as exc:
            return ExecutionResult(
                name=call.name,
                arguments=call.arguments,
                result=None,
                success=False,
                error_message=f"{type(exc).__name__}: {exc}",
            )

    def format_response(self, exec_result: ExecutionResult) -> str:
        """Wrap an execution result in a <tool_response> block ready for re-injection."""
        return f"\n<tool_response>\n{exec_result.to_response_payload()}\n</tool_response>\n"
