"""
probe_evaluator.py
==================
Probe Evaluator for PSCA-SGPO Training

The Probe mechanism directly observes the knowledge state Q_i after each search,
providing fine-grained reward signals for search strategy optimization.

Probe Value Q_i ∈ [0, 1]:
- Q_i = 0: No relevant knowledge acquired
- Q_i = 1: Complete knowledge to answer the question

Search Reward r_i = (Q_i - Q_{i-1}) - λ:
- Positive: Search brought knowledge gain
- Negative: Search was wasteful
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional

from think_retriever.judge.semantic_judge import SemanticJudge, JudgeResult

logger = logging.getLogger(__name__)


@dataclass
class ProbeResult:
    """Result of Probe evaluation."""
    probe_answer: str
    probe_value: float  # Q_i ∈ [0, 1]
    judge_result: Optional[JudgeResult] = None


class ProbeEvaluator:
    """
    Probe Evaluator for PSCA-SGPO training.

    The Probe mechanism works as follows:
    1. After each search, force the model to answer based on current knowledge
    2. Evaluate the answer quality to get Probe Value Q_i
    3. Calculate search reward: r_i = (Q_i - Q_{i-1}) - λ

    Parameters
    ----------
    judge:
        SemanticJudge for answer quality evaluation
    weight:
        Weight for combining multiple signals (default 1.0)

    Example
    -------
    >>> probe_eval = ProbeEvaluator(semantic_judge)
    >>> q_i = probe_eval.evaluate(
    ...     predicted_answer="Paris is the capital of France.",
    ...     reference_answer="Paris",
    ... )
    >>> print(f"Probe Value: {q_i:.3f}")
    Probe Value: 0.875
    """

    def __init__(
        self,
        judge: SemanticJudge,
        weight: float = 1.0,
    ) -> None:
        self.judge = judge
        self.weight = weight

    def evaluate(
        self,
        predicted_answer: str,
        reference_answer: str,
    ) -> float:
        """
        Evaluate the Probe answer quality.

        Parameters
        ----------
        predicted_answer:
            The model's immediate answer after a search
        reference_answer:
            The ground truth answer

        Returns
        -------
        probe_value: float ∈ [0, 1]
            The knowledge state after this search
        """
        if not predicted_answer or not predicted_answer.strip():
            return 0.0

        result = self.judge.judge(
            predicted=predicted_answer,
            reference=reference_answer,
        )

        # Return the semantic similarity score as Probe Value
        return result.score

    def evaluate_batch(
        self,
        predicted_answers: List[str],
        reference_answers: List[str],
    ) -> List[float]:
        """
        Evaluate a batch of Probe answers.

        Parameters
        ----------
        predicted_answers:
            List of predicted answers
        reference_answers:
            List of ground truth answers

        Returns
        -------
        probe_values: List[float]
            List of Probe Values Q_i ∈ [0, 1]
        """
        assert len(predicted_answers) == len(reference_answers), (
            f"Length mismatch: {len(predicted_answers)} vs {len(reference_answers)}"
        )

        probe_values = []
        for pred, ref in zip(predicted_answers, reference_answers):
            q_i = self.evaluate(pred, ref)
            probe_values.append(q_i)

        return probe_values

    @classmethod
    def from_config(cls, cfg: dict) -> "ProbeEvaluator":
        """
        Create ProbeEvaluator from config dict.

        Parameters
        ----------
        cfg:
            Config dict with 'judge' and optionally 'probe' sections
        """
        from think_retriever.judge.semantic_judge import SemanticJudge

        judge = SemanticJudge.from_config(cfg)

        probe_cfg = cfg.get("probe", {})
        weight = probe_cfg.get("weight", 1.0)

        return cls(judge=judge, weight=weight)
