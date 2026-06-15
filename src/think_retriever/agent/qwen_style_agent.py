"""
qwen_style_agent.py
===================
Qwen-Style Function Calling Agent for Two-Stage RL Training

Qwen Format Specification:
- Role tags: <|im_start|>user<|im_end|>, <|im_start|>assistant<|im_end|>
- Tool calls: <tool_call>{"name": "tool_name", "arguments": {...}}</tool_call>
- Tool results: <tool_response>{...}</tool_response>
- Thinking: <think>...</think>

Two-Stage Framework:
- Stage 1: RPA (ReAct Protocol Alignment) - Learn correct protocol flow
- Stage 2: PSCA-SGPO - Learn which search is most effective via Probe mechanism
"""

from __future__ import annotations

import json
import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    StoppingCriteria,
    StoppingCriteriaList,
)

from think_retriever.tools import Calculator, CodeExecutor, ToolRegistry, Verifier
from think_retriever.tools.tool_registry import ExecutionResult

logger = logging.getLogger(__name__)


# Qwen Special Tokens
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_TOOL_RESPONSE_START = "<tool_response>"
_TOOL_RESPONSE_END = "</tool_response>"
_THINK_START = "<think>"
_THINK_END = "</think>"


@dataclass
class ToolCallRecord:
    """Record of a single function invocation during a rollout."""
    name: str
    arguments: Dict[str, Any]
    result: Any
    success: bool
    probe_value: Optional[float] = None  # For PSCA-SGPO: knowledge state after this search
    error_message: Optional[str] = None


@dataclass
class SearchEvent:
    """Record of a search event for PSCA-SGPO training."""
    depth: int  # Search depth (1, 2, 3, ...)
    query: str
    result: str
    probe_value: float = 0.0  # Q_i: knowledge state after this search
    search_reward: float = 0.0  # r_i: search reward = ΔQ - λ


@dataclass
class Episode:
    """Result of a single agent rollout."""
    question: str
    completion: str
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    search_events: List[SearchEvent] = field(default_factory=list)
    tool_usage_counts: Dict[str, int] = field(default_factory=dict)
    num_calls: int = 0
    num_searches: int = 0
    stopped_by: str = "eos"
    token_count: int = 0
    final_probe_value: float = 0.0  # Final knowledge state Q_n


class _StopOnToolCall(StoppingCriteria):
    """Stop generation when a complete tool_call or answer block is generated."""

    def __init__(
        self,
        stop_token_ids: List[List[int]],
        tokenizer: PreTrainedTokenizerBase,
    ) -> None:
        self.stop_token_ids = stop_token_ids
        self.tokenizer = tokenizer

    def __call__(
        self,
        input_ids: torch.LongTensor,
        scores: torch.FloatTensor,
        **kwargs,
    ) -> bool:
        for stop_ids in self.stop_token_ids:
            if len(stop_ids) == 0:
                continue
            if input_ids.shape[1] >= len(stop_ids):
                if input_ids[0, -len(stop_ids):].tolist() == stop_ids:
                    return True
        return False


class QwenStyleAgent:
    """
    Qwen-style function calling agent with two-stage RL training support.

    Key Features:
    1. Qwen Format: <|im_start|>...<|im_end|> role-based conversations
    2. Tool Calling: <tool_call>{"name": "...", "arguments": {...}}</tool_call>
    3. Thinking Support: <think>...</think> for reasoning
    4. Probe Support: Built-in mechanism for PSCA-SGPO training

    Format Example:
    ```
    <|im_start|>user
    What is the capital of France?<|im_end|>
    <|im_start|>assistant
    <think>
    The user is asking about the capital of France. I need to search for this information.
   </think>
    <tool_call>{"name": "search", "arguments": {"query": "capital of France"}}</tool_call><|im_end|>
    <|im_start|>user
    <tool_response>{"result": "Paris is the capital and largest city of France."}</tool_response><|im_end|>
    <|im_start|>assistant
    <think>
    Based on the search result, Paris is the capital of France.
   </think>
    The capital of France is Paris.<|im_end|>
    ```
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        retriever: Any,
        system_prompt: Optional[str] = None,
        max_calls: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        enable_tools: Optional[List[str]] = None,
        probe_evaluator: Optional[Any] = None,  # For PSCA-SGPO
        search_cost: float = 0.05,  # λ in the paper
        early_stop_threshold: float = 0.9,  # τ in the paper
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.max_calls = max_calls
        self.max_new_tokens = max_new_tokens
        self.temperature = temperature
        self.top_p = top_p

        # Probe-related settings for PSCA-SGPO
        self.probe_evaluator = probe_evaluator
        self.search_cost = search_cost  # λ
        self.early_stop_threshold = early_stop_threshold  # τ

        # Initialize tool registry
        self.tool_registry = ToolRegistry()
        self._register_builtin_tools(enable_tools or ["search"])

        # Build system prompt
        self.system_prompt = system_prompt or self._build_qwen_system_prompt()

        # Pre-tokenise stop strings
        self._stop_token_ids: List[List[int]] = []
        for stop_str in [_TOOL_CALL_END, _IM_END]:
            for variant in [stop_str, " " + stop_str, "\n" + stop_str]:
                token_ids = tokenizer.encode(variant, add_special_tokens=False)
                if token_ids:
                    self._stop_token_ids.append(token_ids)

        # Compile regex patterns for parsing
        self._tool_call_pattern = re.compile(
            r'<tool_call>\s*(\{.*?\})\s*</tool_call>', re.DOTALL
        )
        self._think_pattern = re.compile(r'<think>(.*?)</think>', re.DOTALL)

    def _build_qwen_system_prompt(self) -> str:
        """Build Qwen-style system prompt with tool definitions."""
        tools_desc = self.tool_registry.render_schemas_for_prompt()

        return f"""You are a helpful AI assistant with access to tools.
Always follow the EXACT output structure below.

When you need to use a tool, output:
<tool_call>{{"name": "tool_name", "arguments": {{"param1": "value1", ...}}}}</tool_call>

Available tools:
{tools_desc}

Required output pattern — follow this order exactly:

  Step 1. <think>Reason about what the user asked. Decide whether a tool
          is needed, and if so, which tool with what arguments.</think>

  Step 2. <tool_call>{{"name": "...", "arguments": {{...}}}}</tool_call>
          [the system injects <tool_response>...</tool_response>]

  Step 3. <think>Read the tool result. Decide whether you need another
          tool call or have enough information to answer.</think>

  Step 4. Repeat steps 2-3 as needed.

  Step 5. <think>Compile your final answer based on all the information
          gathered.</think>

  Step 6. <answer>Write your final natural-language answer here.</answer>

Rules:
• Use <think>...</think> BEFORE every <tool_call>
• Use <think>...</think> right BEFORE your final <answer>
• Wrap your final answer in <answer>...</answer> tags
• Produce exactly ONE <answer> block, placed at the very end
• Do NOT put any text after </answer>
• Each <tool_call> should be on its own line

Example — User: "What is 127 * 42?"

<think>The user wants me to compute 127 * 42.
This is a math problem so I should use the calculator tool.</think>
<tool_call>{{"name": "calculator", "arguments": {{"expression": "127 * 42"}}}}</tool_call>
<tool_response>{{"success": true, "result": "5334"}}</tool_response>
<think>The calculator returned 5334. That is the final answer.</think>
<answer>127 × 42 = 5334</answer>
"""

    def _register_builtin_tools(self, enable_tools: List[str]) -> None:
        """Register the built-in tools with Qwen-compatible JSON schemas."""
        if "search" in enable_tools:
            if self.retriever is None:
                logger.warning("search requested but retriever is None; skipping.")
            else:
                self.tool_registry.register_function(
                    name="search",
                    description="Search for relevant information from the document corpus.",
                    parameters={
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Search query in natural language.",
                            }
                        },
                        "required": ["query"],
                    },
                    callable=lambda **kw: self.retriever.search(kw["query"]),
                )

        if "calculator" in enable_tools:
            calc = Calculator()
            self.tool_registry.register_function(
                name="calculator",
                description="Evaluate an arithmetic expression and return the numeric result.",
                parameters={
                    "type": "object",
                    "properties": {
                        "expression": {
                            "type": "string",
                            "description": "Arithmetic expression (e.g. '2*(3+4)').",
                        }
                    },
                    "required": ["expression"],
                },
                callable=lambda **kw: calc.execute(kw["expression"]),
            )

        if "code" in enable_tools:
            executor = CodeExecutor()
            self.tool_registry.register_function(
                name="code",
                description="Execute Python code and return the output.",
                parameters={
                    "type": "object",
                    "properties": {
                        "code": {
                            "type": "string",
                            "description": "Python source code to execute.",
                        }
                    },
                    "required": ["code"],
                },
                callable=lambda **kw: executor.execute(kw["code"]),
            )

        if "verify" in enable_tools:
            verifier = Verifier(retriever=self.retriever)
            self.tool_registry.register_function(
                name="verify",
                description="Verify a factual claim against the document corpus.",
                parameters={
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "Claim to fact-check.",
                        }
                    },
                    "required": ["claim"],
                },
                callable=lambda **kw: verifier.execute(kw["claim"]),
            )

    def _build_messages(
        self, question: str, history: Optional[List[Dict[str, str]]] = None
    ) -> List[Dict[str, str]]:
        """Build message list for chat template."""
        messages = [{"role": "system", "content": self.system_prompt}]

        if history:
            messages.extend(history)

        messages.append({"role": "user", "content": question})

        return messages

    def _apply_chat_template(self, messages: List[Dict[str, str]]) -> str:
        """Apply Qwen chat template to messages."""
        if hasattr(self.tokenizer, "apply_chat_template"):
            return self.tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        # Fallback: manual template
        text = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            text += f"{_IM_START}{role}\n{content}{_IM_END}\n"
        text += f"{_IM_START}assistant\n"
        return text

    def _parse_tool_calls(self, text: str) -> List[Dict[str, Any]]:
        """Parse tool calls from generated text."""
        matches = self._tool_call_pattern.findall(text)
        tool_calls = []
        for match in matches:
            try:
                payload = json.loads(match)
                if isinstance(payload, dict) and "name" in payload:
                    tool_calls.append(payload)
            except json.JSONDecodeError:
                continue
        return tool_calls

    def _format_tool_response(self, result: Any, success: bool = True) -> str:
        """Format tool response in Qwen style."""
        response_data = {
            "success": success,
            "result": result if success else str(result),
        }
        return f"{_TOOL_RESPONSE_START}{json.dumps(response_data)}{_TOOL_RESPONSE_END}"

    @torch.inference_mode()
    def rollout(
        self,
        question: str,
        ground_truth: Optional[str] = None,
        enable_probe: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> Episode:
        """
        Single episode rollout with optional Probe mechanism for PSCA-SGPO.

        Parameters
        ----------
        question: The question to answer
        ground_truth: For Probe evaluation (optional)
        enable_probe: If True, enable Probe mechanism for PSCA-SGPO
        temperature, top_p: Sampling parameters
        """
        temp = temperature if temperature is not None else self.temperature
        topp = top_p if top_p is not None else self.top_p

        messages = self._build_messages(question)
        working_text = self._apply_chat_template(messages)
        generated_text = ""

        executed_calls: List[ToolCallRecord] = []
        search_events: List[SearchEvent] = []
        usage_counts: Dict[str, int] = {}
        stopped_by = "eos"
        search_depth = 0
        prev_probe_value = 0.0  # Q_0 = 0

        stopping = StoppingCriteriaList([_StopOnToolCall(self._stop_token_ids, self.tokenizer)])

        for step in range(self.max_calls + 1):
            # Tokenize current context
            input_ids = self.tokenizer(
                working_text,
                return_tensors="pt",
                truncation=True,
                max_length=8192,
            ).input_ids.to(self.model.device)

            # Generate next tokens
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_new_tokens,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_p=topp if temp > 0 else None,
                stopping_criteria=stopping,
                pad_token_id=self.tokenizer.eos_token_id,
            )

            new_ids = output_ids[0, input_ids.shape[1]:]
            new_text = self.tokenizer.decode(new_ids, skip_special_tokens=False)
            generated_text += new_text
            working_text += new_text

            # Check if generation ended
            if not new_text.rstrip().endswith(_TOOL_CALL_END):
                stopped_by = "eos"
                break

            # Parse and execute tool calls
            tool_calls = self._parse_tool_calls(generated_text)
            if not tool_calls:
                stopped_by = "eos"
                break

            for call in tool_calls[-1:]:  # Execute only the latest call
                tool_name = call.get("name")
                arguments = call.get("arguments", {})

                # Execute tool
                exec_result = self.tool_registry.execute(
                    name=tool_name,
                    arguments=arguments,
                )

                record = ToolCallRecord(
                    name=tool_name,
                    arguments=arguments,
                    result=exec_result.result,
                    success=exec_result.success,
                    error_message=exec_result.error_message,
                )
                executed_calls.append(record)
                usage_counts[tool_name] = usage_counts.get(tool_name, 0) + 1

                # Format and inject response
                response_block = self._format_tool_response(
                    exec_result.result,
                    exec_result.success,
                )
                generated_text += response_block
                working_text += response_block

                # PSCA-SGPO: Probe mechanism for search tools
                if enable_probe and tool_name == "search" and self.probe_evaluator:
                    search_depth += 1

                    # Build context for probe
                    probe_context = working_text.replace(
                        _TOOL_RESPONSE_START + json.dumps({
                            "success": True,
                            "result": exec_result.result
                        }) + _TOOL_RESPONSE_END,
                        ""
                    )

                    # Extract current answer (before adding response to model)
                    # This simulates the Probe: model must answer immediately
                    probe_answer = self._generate_probe_answer(
                        question,
                        exec_result.result,
                        max_tokens=50,
                    )

                    # Evaluate probe
                    if ground_truth and probe_answer:
                        probe_result = self.probe_evaluator.evaluate(
                            probe_answer,
                            ground_truth,
                        )
                        current_probe_value = probe_result  # Q_i ∈ [0, 1]
                    else:
                        current_probe_value = prev_probe_value

                    # Calculate search reward
                    delta_q = current_probe_value - prev_probe_value
                    search_reward = delta_q - self.search_cost  # r_i = ΔQ - λ

                    search_event = SearchEvent(
                        depth=search_depth,
                        query=arguments.get("query", ""),
                        result=exec_result.result,
                        probe_value=current_probe_value,
                        search_reward=search_reward,
                    )
                    search_events.append(search_event)
                    record.probe_value = current_probe_value

                    # Early stopping check
                    if current_probe_value >= self.early_stop_threshold:
                        # Add bonus reward for efficient search
                        search_event.search_reward += 0.1 * (self.max_calls - search_depth)
                        logger.debug(
                            f"Early stop at depth {search_depth}, "
                            f"Q={current_probe_value:.3f}"
                        )

                    prev_probe_value = current_probe_value

            if len(executed_calls) >= self.max_calls:
                stopped_by = "max_calls"
                break
        else:
            stopped_by = "max_calls"

        # Token count
        token_count = self.tokenizer(
            generated_text, return_tensors="pt"
        ).input_ids.shape[1]

        return Episode(
            question=question,
            completion=generated_text,
            tool_calls=executed_calls,
            search_events=search_events,
            tool_usage_counts=usage_counts,
            num_calls=len(executed_calls),
            num_searches=search_depth,
            stopped_by=stopped_by,
            token_count=token_count,
            final_probe_value=prev_probe_value,
        )

    def _generate_probe_answer(
        self,
        question: str,
        context: str,
        max_tokens: int = 50,
    ) -> Optional[str]:
        """
        Generate a probe answer for PSCA-SGPO.

        This is called after each search to evaluate the knowledge state.
        The model is forced to answer based on current context.
        """
        if not self.probe_evaluator:
            return None

        probe_messages = [
            {"role": "system", "content": "Answer the question based ONLY on the provided context."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"},
        ]

        prompt = self._apply_chat_template(probe_messages)

        input_ids = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048,
        ).input_ids.to(self.model.device)

        output_ids = self.model.generate(
            input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,
            pad_token_id=self.tokenizer.eos_token_id,
        )

        answer_ids = output_ids[0, input_ids.shape[1]:]
        answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        # Extract just the answer part (before <|im_end|>)
        if _IM_END in answer:
            answer = answer.split(_IM_END)[0].strip()

        return answer if answer else None

    def rollout_group(
        self,
        question: str,
        ground_truth: Optional[str],
        group_size: int,
        enable_probe: bool = False,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
    ) -> List[Episode]:
        """Generate group_size independent episodes for GRPO."""
        return [
            self.rollout(
                question,
                ground_truth=ground_truth,
                enable_probe=enable_probe,
                temperature=temperature,
                top_p=top_p,
            )
            for _ in range(group_size)
        ]

    @classmethod
    def from_pretrained(
        cls,
        model_name_or_path: str,
        retriever: Any,
        system_prompt: Optional[str] = None,
        *,
        torch_dtype=torch.bfloat16,
        device_map: str = "auto",
        attn_implementation: str = "flash_attention_2",
        max_calls: int = 5,
        max_new_tokens: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.95,
        enable_tools: Optional[List[str]] = None,
        probe_evaluator: Optional[Any] = None,
        search_cost: float = 0.05,
        early_stop_threshold: float = 0.9,
    ) -> "QwenStyleAgent":
        """Factory method to create agent from pretrained model."""
        logger.info("Loading model: %s", model_name_or_path)
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path, trust_remote_code=True
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map=device_map,
            attn_implementation=attn_implementation,
            trust_remote_code=True,
        )
        model.eval()

        return cls(
            model=model,
            tokenizer=tokenizer,
            retriever=retriever,
            system_prompt=system_prompt,
            max_calls=max_calls,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            enable_tools=enable_tools,
            probe_evaluator=probe_evaluator,
            search_cost=search_cost,
            early_stop_threshold=early_stop_threshold,
        )
