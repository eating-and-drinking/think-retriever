"""
two_stage_reward.py
===================
Two-Stage Reward Function for RPA + PSCA-SGPO Training

Stage 1: RPA (ReAct Protocol Alignment)
- Format reward: XML structure correctness
- Protocol reward: Think→Search→Content→Answer flow
- Budget reward: Search count penalty
- Answer reward: Final answer quality

Stage 2: PSCA-SGPO (Probe-based Search Credit Assignment)
- Search reward: r_i = (Q_i - Q_{i-1}) - λ
- State-aware advantage normalization
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from think_retriever.judge.semantic_judge import SemanticJudge, JudgeResult

logger = logging.getLogger(__name__)


# ── Stage 1: RPA Rewards ──────────────────────────────────────────────────────


@dataclass
class FormatRewardResult:
    """Result of format reward computation."""
    passed: bool
    reward: float
    reason: str
    has_think: bool = False
    has_tool_call: bool = False
    has_tool_response: bool = False
    has_answer: bool = False


@dataclass
class ProtocolRewardResult:
    """Result of protocol reward computation."""
    reward: float
    reason: str
    search_after_think: bool = False
    content_after_search: bool = False
    answer_after_content: bool = False


@dataclass
class BudgetRewardResult:
    """Result of budget reward computation."""
    reward: float
    reason: str
    num_searches: int = 0
    penalty: float = 0.0


@dataclass
class AnswerRewardResult:
    """Result of answer reward computation."""
    reward: float
    reason: str
    is_correct: bool = False
    judge_result: Optional[JudgeResult] = None


# ── Stage 2: PSCA-SGPO Rewards ────────────────────────────────────────────────


@dataclass
class SearchRewardResult:
    """Result of search reward computation."""
    depth: int
    q_value: float  # Probe Value Q_i
    search_reward: float  # r_i = ΔQ - λ
    delta_q: float  # Knowledge gain from this search


@dataclass
class TwoStageRewardBreakdown:
    """Complete breakdown of two-stage rewards."""
    # Stage 1: RPA
    format_result: Optional[FormatRewardResult] = None
    protocol_result: Optional[ProtocolRewardResult] = None
    budget_result: Optional[BudgetRewardResult] = None
    answer_result: Optional[AnswerRewardResult] = None

    # Stage 2: PSCA-SGPO
    search_results: List[SearchRewardResult] = field(default_factory=list)

    # Aggregated rewards
    stage1_reward: float = 0.0  # RPA total
    stage2_reward: float = 0.0  # PSCA-SGPO total
    total_reward: float = 0.0


# ── Stage 1: RPA Reward Functions ──────────────────────────────────────────────


def compute_format_reward(
    completion: str,
    reward_correct: float = 1.0,
    reward_wrong: float = -1.0,
) -> FormatRewardResult:
    """
    Compute format reward for Qwen-style XML structure.

    Phase 1 — hard constraints (SFT should teach these):
      1. <think> / <tool_call> / <tool_response> tags are properly paired.

    Phase 2 — soft (graded) answer reward (learned during RL):
      2. <answer> tags present and well-formed → +1.0
         plain-text answer after tool responses   → +0.4
         no answer content at all                   → -0.5
      3. <think> immediately before <answer>       → no penalty;
                                                      missing → -0.25
    """
    # ── Phase 1: hard tag pairing ─────────────────────────────
    think_open = len(re.findall(r'<think>', completion))
    think_close = len(re.findall(r'</think>', completion))
    tool_call_open = len(re.findall(r'<tool_call>', completion))
    tool_call_close = len(re.findall(r'</tool_call>', completion))
    tool_resp_open = len(re.findall(r'<tool_response>', completion))
    tool_resp_close = len(re.findall(r'</tool_response>', completion))

    if think_open != think_close:
        return FormatRewardResult(
            passed=False,
            reward=reward_wrong,
            reason=f"Unmatched <think> tags: {think_open} open vs {think_close} close",
        )
    if tool_call_open != tool_call_close:
        return FormatRewardResult(
            passed=False,
            reward=reward_wrong,
            reason=f"Unmatched <tool_call> tags: {tool_call_open} open vs {tool_call_close} close",
        )
    if tool_resp_open != tool_resp_close:
        return FormatRewardResult(
            passed=False,
            reward=reward_wrong,
            reason=f"Unmatched <tool_response> tags: {tool_resp_open} open vs {tool_resp_close} close",
        )

    # ── Phase 2: answer format (soft) ─────────────────────────
    answer_open = len(re.findall(r'<answer>', completion))
    answer_close = len(re.findall(r'</answer>', completion))

    base_score = 1.0
    reasons: List[str] = []
    has_answer_tag = False

    if answer_open > 0 and answer_open == answer_close:
        # Valid <answer> tag pair
        inner_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
        if inner_match and inner_match.group(1).strip():
            has_answer_tag = True
            reasons.append("<answer> tag present & non-empty")
            if answer_open == 1:
                reasons.append("exactly one <answer>")
            else:
                base_score -= 0.2
                reasons.append(f"{answer_open} <answer> blocks (-0.2)")

            # Check for <think> immediately before <answer>
            answer_start_pos = completion.find("<answer>")
            think_close_positions = [
                m.end() for m in re.finditer(r"</think>", completion)
            ]
            has_think_before = False
            for tc in reversed(think_close_positions):
                if tc <= answer_start_pos:
                    gap = completion[tc:answer_start_pos].strip()
                    if len(gap) < 30:
                        has_think_before = True
                    break
            if not has_think_before and think_open > 0:
                base_score -= 0.25
                reasons.append("no <think> before <answer> (-0.25)")
            elif has_think_before:
                reasons.append("<think> before <answer>")
        else:
            base_score -= 0.5
            reasons.append("empty <answer> block (-0.5)")
    else:
        # No <answer> tags — check for plain-text answer
        has_plain_answer = False
        resp_matches = list(re.finditer(r"</tool_response>", completion))
        if resp_matches:
            tail = completion[resp_matches[-1].end():]
            tail = re.sub(r"<[^>]+>", "", tail).strip()
            if tail and len(tail) > 2:
                has_plain_answer = True
        if not has_plain_answer and tool_call_open == 0:
            cleaned = re.sub(r"<[^>]+>", "", completion).strip()
            if len(cleaned) > 2:
                has_plain_answer = True

        if has_plain_answer:
            base_score = 0.4
            reasons.append("no <answer> tag but plain-text answer (+0.4)")
        elif tool_call_open > 0:
            base_score -= 0.5
            reasons.append("no answer after tool calls (-0.5)")
        else:
            base_score = 0.2
            reasons.append("no tool calls, no answer (+0.2)")

    final_reward = max(reward_wrong, min(reward_correct, base_score))

    return FormatRewardResult(
        passed=final_reward >= 0.8,
        reward=final_reward,
        reason="; ".join(reasons),
        has_think=think_open > 0,
        has_tool_call=tool_call_open > 0,
        has_tool_response=tool_resp_open > 0,
        has_answer=has_answer_tag or answer_open > 0,
    )


def compute_protocol_reward(
    completion: str,
    reward_full: float = 1.0,
    reward_partial: float = 0.5,
    reward_none: float = 0.0,
) -> ProtocolRewardResult:
    """
    Compute protocol reward for ReAct flow.

    Checks:
    1. Search happens after Think (reasoning before action)
    2. Content/Response comes after Search
    3. Answer comes after Content
    """
    # Find positions
    think_positions = [m.start() for m in re.finditer(r'<think>', completion)]
    search_positions = [m.start() for m in re.finditer(r'<tool_call>', completion)]
    response_positions = [m.start() for m in re.finditer(r'<tool_response>', completion)]

    # Check: search_after_think
    search_after_think = False
    if think_positions and search_positions:
        # Check if at least one search comes after a think
        for search_pos in search_positions:
            for think_pos in think_positions:
                if think_pos < search_pos:
                    search_after_think = True
                    break
            if search_after_think:
                break

    # Check: content_after_search
    content_after_search = False
    if search_positions and response_positions:
        for resp_pos in response_positions:
            for search_pos in search_positions:
                if search_pos < resp_pos:
                    content_after_search = True
                    break
            if content_after_search:
                break

    # Check: answer_after_content
    answer_after_content = False
    end_positions = [m.start() for m in re.finditer(r'<|im_end|>', completion)]
    if response_positions and end_positions:
        for end_pos in end_positions:
            for resp_pos in response_positions:
                if resp_pos < end_pos:
                    answer_after_content = True
                    break
            if answer_after_content:
                break

    # Compute reward
    protocol_score = sum([
        search_after_think,
        content_after_search,
        answer_after_content,
    ]) / 3.0

    if protocol_score == 1.0:
        reward = reward_full
    elif protocol_score > 0:
        reward = reward_partial
    else:
        reward = reward_none

    return ProtocolRewardResult(
        reward=reward,
        reason=f"Protocol score: {protocol_score:.2f}",
        search_after_think=search_after_think,
        content_after_search=content_after_search,
        answer_after_content=answer_after_content,
    )


def compute_budget_reward(
    num_searches: int,
    max_searches: int = 5,
    cost_per_search: float = -0.1,
    bonus_for_correct: float = 0.5,
    is_correct: bool = False,
) -> BudgetRewardResult:
    """
    Compute budget reward for search count.

    Formula:
        budget_reward = num_searches * cost_per_search + (bonus if correct)
    """
    penalty = num_searches * cost_per_search
    bonus = bonus_for_correct if is_correct else 0.0

    # Cap penalty for excessive searches
    if num_searches > max_searches:
        penalty *= 2  # Double penalty for exceeding max

    reward = penalty + bonus

    return BudgetRewardResult(
        reward=reward,
        reason=f"{num_searches} searches × {cost_per_search} + {'bonus' if is_correct else 'no bonus'}",
        num_searches=num_searches,
        penalty=penalty,
    )


def compute_answer_reward(
    predicted_answer: Optional[str],
    reference_answer: str,
    judge: SemanticJudge,
    reward_correct: float = 1.0,
    reward_wrong: float = 0.0,
) -> AnswerRewardResult:
    """
    Compute answer reward for final answer quality.
    """
    if not predicted_answer or not predicted_answer.strip():
        return AnswerRewardResult(
            reward=reward_wrong,
            reason="No answer extracted",
            is_correct=False,
        )

    judge_result = judge.judge(
        predicted=predicted_answer,
        reference=reference_answer,
    )

    is_correct = judge_result.is_equivalent
    reward = reward_correct if is_correct else reward_wrong

    return AnswerRewardResult(
        reward=reward,
        reason=f"Answer {'correct' if is_correct else 'wrong'} (score={judge_result.score:.3f})",
        is_correct=is_correct,
        judge_result=judge_result,
    )


# ── Stage 2: PSCA-SGPO Reward Functions ───────────────────────────────────────


def compute_search_reward(
    q_value: float,
    q_prev: float,
    search_cost: float = 0.05,
) -> SearchRewardResult:
    """
    Compute search reward for PSCA-SGPO.

    Formula:
        r_i = (Q_i - Q_{i-1}) - λ

    where:
        Q_i: Current Probe Value
        Q_{i-1}: Previous Probe Value
        λ: Search cost (default 0.05)
    """
    delta_q = q_value - q_prev
    search_reward = delta_q - search_cost

    return SearchRewardResult(
        depth=0,  # Will be set by caller
        q_value=q_value,
        search_reward=search_reward,
        delta_q=delta_q,
    )


def extract_final_answer(completion: str) -> Optional[str]:
    """
    Extract the final answer from Qwen-style completion.

    Strategies (in order of preference):
    1. Find content inside <answer>...</answer> tags (canonical format)
    2. Find text after last <|im_end|>
    3. Remove common prefixes like "The answer is"
    """
    # 1. Prefer <answer> tags (canonical format)
    answer_match = re.search(r"<answer>(.*?)</answer>", completion, re.DOTALL)
    if answer_match:
        answer = answer_match.group(1).strip()
        if answer:
            return answer

    # 2. Find the last assistant turn
    matches = list(re.finditer(r'<\|im_end\|>', completion))

    if not matches:
        # Fallback: use the entire completion
        answer = completion.strip()
    else:
        # Get text after the last im_end
        last_end = matches[-1].end()
        answer = completion[last_end:].strip()

    # Clean up the answer
    answer = re.sub(r'<think>.*?</think>', '', answer, flags=re.DOTALL)  # Remove thinking
    answer = re.sub(r'<tool_call>.*?</tool_call>', '', answer, flags=re.DOTALL)  # Remove tool calls
    answer = answer.strip()

    # Remove common prefixes
    prefixes_to_remove = [
        "the answer is",
        "answer:",
        "the final answer is",
        "in conclusion,",
    ]
    answer_lower = answer.lower()
    for prefix in prefixes_to_remove:
        if answer_lower.startswith(prefix):
            answer = answer[len(prefix):].strip()

    return answer if answer else None


# ── Two-Stage Reward Orchestrator ──────────────────────────────────────────────


class TwoStageRewardFn:
    """
    Two-Stage Reward Function for RPA + PSCA-SGPO training.

    Stage 1: RPA (ReAct Protocol Alignment)
        - Format reward
        - Protocol reward
        - Budget reward
        - Answer reward

    Stage 2: PSCA-SGPO (Probe-based Search Credit Assignment)
        - Search rewards computed from Probe Values
        - State-aware advantage normalization done separately
    """

    def __init__(
        self,
        judge: SemanticJudge,
        # Stage 1 rewards
        format_correct: float = 1.0,
        format_wrong: float = -1.0,
        protocol_full: float = 1.0,
        protocol_partial: float = 0.5,
        cost_per_search: float = -0.1,
        bonus_for_correct: float = 0.5,
        answer_correct: float = 1.0,
        answer_wrong: float = 0.0,
        # Stage 2 rewards
        search_cost: float = 0.05,
    ) -> None:
        self.judge = judge
        self.format_correct = format_correct
        self.format_wrong = format_wrong
        self.protocol_full = protocol_full
        self.protocol_partial = protocol_partial
        self.cost_per_search = cost_per_search
        self.bonus_for_correct = bonus_for_correct
        self.answer_correct = answer_correct
        self.answer_wrong = answer_wrong
        self.search_cost = search_cost

    @classmethod
    def from_config(cls, cfg: dict) -> "TwoStageRewardFn":
        """Create from config dict."""
        from think_retriever.judge.semantic_judge import SemanticJudge

        judge = SemanticJudge.from_config(cfg)

        r1 = cfg.get("reward", {})
        r2 = cfg.get("psca_sgpo", {})

        return cls(
            judge=judge,
            format_correct=r1.get("format_correct", 1.0),
            format_wrong=r1.get("format_wrong", -1.0),
            protocol_full=r1.get("protocol_full", 1.0),
            protocol_partial=r1.get("protocol_partial", 0.5),
            cost_per_search=r1.get("cost_per_search", -0.1),
            bonus_for_correct=r1.get("bonus_for_correct", 0.5),
            answer_correct=r1.get("answer_correct", 1.0),
            answer_wrong=r1.get("answer_wrong", 0.0),
            search_cost=r2.get("search_cost", 0.05),
        )

    def compute_stage1(
        self,
        completion: str,
        reference_answer: str,
        num_searches: int,
        max_searches: int = 5,
    ) -> TwoStageRewardBreakdown:
        """
        Compute Stage 1 (RPA) rewards.

        Returns
        -------
        breakdown: TwoStageRewardBreakdown
            Breakdown of all Stage 1 rewards
        """
        # Format reward
        format_result = compute_format_reward(
            completion,
            reward_correct=self.format_correct,
            reward_wrong=self.format_wrong,
        )

        # Protocol reward
        protocol_result = compute_protocol_reward(
            completion,
            reward_full=self.protocol_full,
            reward_partial=self.protocol_partial,
        )

        # Answer reward (for budget calculation)
        predicted_answer = extract_final_answer(completion)
        answer_result = compute_answer_reward(
            predicted_answer,
            reference_answer,
            self.judge,
            reward_correct=self.answer_correct,
            reward_wrong=self.answer_wrong,
        )

        # Budget reward
        budget_result = compute_budget_reward(
            num_searches=num_searches,
            max_searches=max_searches,
            cost_per_search=self.cost_per_search,
            bonus_for_correct=self.bonus_for_correct,
            is_correct=answer_result.is_correct,
        )

        # Aggregate Stage 1 reward
        stage1_reward = (
            format_result.reward
            + protocol_result.reward
            + answer_result.reward
            + budget_result.reward
        )

        return TwoStageRewardBreakdown(
            format_result=format_result,
            protocol_result=protocol_result,
            answer_result=answer_result,
            budget_result=budget_result,
            stage1_reward=stage1_reward,
            total_reward=stage1_reward,
        )

    def compute_stage2_from_episodes(
        self,
        episodes: List[Any],  # List of Episode objects
        reference_answer: str,
    ) -> TwoStageRewardBreakdown:
        """
        Compute Stage 2 (PSCA-SGPO) rewards from episodes.

        This method extracts Probe Values and Search Rewards from episodes
        collected during rollout.

        Parameters
        ----------
        episodes:
            List of Episode objects with search_events containing
            probe_value and search_reward fields
        reference_answer:
            Ground truth answer

        Returns
        -------
        breakdown: TwoStageRewardBreakdown
            Breakdown with Stage 2 rewards
        """
        search_results: List[SearchRewardResult] = []

        for episode in episodes:
            if not hasattr(episode, 'search_events'):
                continue

            prev_q = 0.0  # Q_0 = 0
            for event in episode.search_events:
                # Recompute search reward if not already set
                if event.search_reward == 0.0:
                    delta_q = event.probe_value - prev_q
                    search_reward = delta_q - self.search_cost
                else:
                    search_reward = event.search_reward
                    delta_q = event.probe_value - prev_q

                result = SearchRewardResult(
                    depth=event.depth,
                    q_value=event.probe_value,
                    search_reward=search_reward,
                    delta_q=delta_q,
                )
                search_results.append(result)

                prev_q = event.probe_value

        # Aggregate Stage 2 reward
        stage2_reward = sum(r.search_reward for r in search_results)

        return TwoStageRewardBreakdown(
            search_results=search_results,
            stage2_reward=stage2_reward,
            total_reward=stage2_reward,
        )

    def compute_batch(
        self,
        completions: List[str],
        reference_answers: List[str],
        num_searches_list: List[int],
        episodes: Optional[List[Any]] = None,
        stage: int = 1,
    ) -> tuple[List[float], List[TwoStageRewardBreakdown]]:
        """
        Compute rewards for a batch of completions.

        Parameters
        ----------
        completions:
            List of model completions
        reference_answers:
            List of ground truth answers
        num_searches_list:
            List of search counts per completion
        episodes:
            List of Episode objects (for Stage 2)
        stage:
            Training stage (1 or 2)

        Returns
        -------
        (rewards, breakdowns)
        """
        assert len(completions) == len(reference_answers)
        assert len(completions) == len(num_searches_list)

        rewards = []
        breakdowns = []

        for i, (completion, ref_answer, num_searches) in enumerate(
            zip(completions, reference_answers, num_searches_list)
        ):
            if stage == 1:
                bd = self.compute_stage1(
                    completion, ref_answer, num_searches
                )
            else:
                # Stage 2: use episodes if provided
                if episodes and i < len(episodes):
                    bd = self.compute_stage2_from_episodes(
                        [episodes[i]], ref_answer
                    )
                else:
                    bd = TwoStageRewardBreakdown()

            rewards.append(bd.total_reward)
            breakdowns.append(bd)

        return rewards, breakdowns

    def summarize_batch(self, breakdowns: List[TwoStageRewardBreakdown]) -> dict:
        """Return aggregate statistics for logging."""
        n = len(breakdowns)
        if n == 0:
            return {}

        return {
            "reward/mean": sum(b.total_reward for b in breakdowns) / n,
            "reward/stage1_mean": sum(b.stage1_reward for b in breakdowns) / n,
            "reward/stage2_mean": sum(b.stage2_reward for b in breakdowns) / n,
            "reward/format_pass_rate": sum(
                1 for b in breakdowns
                if b.format_result and b.format_result.passed
            ) / n,
            "reward/answer_accuracy": sum(
                1 for b in breakdowns
                if b.answer_result and b.answer_result.is_correct
            ) / n,
        }
