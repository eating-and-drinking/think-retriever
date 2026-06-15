"""
rewards/__init__.py
───────────────────
Two-stage reward functions for RPA + PSCA-SGPO training.

Stage 1 (RPA):
  * format     — XML structure correctness (±1)
  * protocol   — Think→Search→Content→Answer flow (0-1)
  * budget     — penalty per extra search (<= 0)
  * answer     — semantic similarity ([0, 1])

Stage 2 (PSCA-SGPO):
  * search     — knowledge gain minus search cost (ΔQ - λ)
"""

from agentic_rag.rewards.two_stage_reward import (
    TwoStageRewardFn,
    TwoStageRewardBreakdown,
    FormatRewardResult,
    ProtocolRewardResult,
    BudgetRewardResult,
    AnswerRewardResult,
    SearchRewardResult,
    compute_format_reward,
    compute_protocol_reward,
    compute_budget_reward,
    compute_answer_reward,
    compute_search_reward,
    extract_final_answer,
)

__all__ = [
    "TwoStageRewardFn",
    "TwoStageRewardBreakdown",
    "FormatRewardResult",
    "ProtocolRewardResult",
    "BudgetRewardResult",
    "AnswerRewardResult",
    "SearchRewardResult",
    "compute_format_reward",
    "compute_protocol_reward",
    "compute_budget_reward",
    "compute_answer_reward",
    "compute_search_reward",
    "extract_final_answer",
]