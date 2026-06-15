"""
semantic_judge.py
─────────────────
SemanticJudge — Hybrid-voting semantic equivalence checker.

Instead of brittle exact-string matching, this module combines five
complementary signals via a weighted vote to determine whether a
predicted answer is semantically equivalent to the ground truth.

Signals & weights (configurable)
─────────────────────────────────
  EM     (Exact Match, normalised)          default weight: 0.20
  ROUGE  (ROUGE-L F1)                       default weight: 0.25
  BLEU   (BLEU-4 sentence score)            default weight: 0.15
  FUZZY  (RapidFuzz token_sort_ratio / 100) default weight: 0.20
  SEM    (Sentence-Transformer cosine sim)  default weight: 0.20

Weighted score = Σ weight_i × signal_i

Decision: equivalent  iff  weighted_score ≥ threshold  (default 0.60)

All normalisation/lower-casing is applied before scoring to reduce
surface noise while preserving semantic content.
"""

from __future__ import annotations

import logging
import re
import string
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Dict, List, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)

# ── Optional heavy imports (fail gracefully so unit tests run without GPU) ────
try:
    from rouge_score import rouge_scorer as _rouge_scorer_mod

    _ROUGE_AVAILABLE = True
except ImportError:
    _ROUGE_AVAILABLE = False
    logger.warning("rouge_score not installed; ROUGE signal disabled.")

try:
    import sacrebleu as _sacrebleu

    _BLEU_AVAILABLE = True
except ImportError:
    _BLEU_AVAILABLE = False
    logger.warning("sacrebleu not installed; BLEU signal disabled.")

try:
    from rapidfuzz import fuzz as _fuzz

    _FUZZY_AVAILABLE = True
except ImportError:
    _FUZZY_AVAILABLE = False
    logger.warning("rapidfuzz not installed; fuzzy signal disabled.")

try:
    from sentence_transformers import SentenceTransformer as _SentenceTransformer

    _SEMANTIC_AVAILABLE = True
except ImportError:
    _SEMANTIC_AVAILABLE = False
    logger.warning("sentence-transformers not installed; semantic signal disabled.")


# ── Result dataclass ──────────────────────────────────────────────────────────


@dataclass
class JudgeResult:
    """Detailed result from SemanticJudge."""

    predicted: str
    reference: str
    is_equivalent: bool
    score: float  # final weighted score

    # Per-signal scores (None if signal unavailable / disabled)
    em_score: Optional[float] = None
    rouge_score: Optional[float] = None
    bleu_score: Optional[float] = None
    fuzzy_score: Optional[float] = None
    semantic_score: Optional[float] = None

    signal_weights: Dict[str, float] = field(default_factory=dict)
    threshold: float = 0.60

    def to_dict(self) -> dict:
        return {
            "is_equivalent": self.is_equivalent,
            "score": round(self.score, 4),
            "em_score": self.em_score,
            "rouge_score": self.rouge_score,
            "bleu_score": self.bleu_score,
            "fuzzy_score": self.fuzzy_score,
            "semantic_score": self.semantic_score,
        }


# ── Main class ────────────────────────────────────────────────────────────────


class SemanticJudge:
    """
    Hybrid-voting semantic equivalence judge.

    Parameters
    ----------
    em_weight:
        Weight for exact-match signal.
    rouge_weight:
        Weight for ROUGE-L F1.
    bleu_weight:
        Weight for BLEU-4.
    fuzzy_weight:
        Weight for RapidFuzz token_sort_ratio.
    semantic_weight:
        Weight for sentence-transformer cosine similarity.
    threshold:
        Score ≥ threshold → equivalent.
    semantic_model_name:
        HuggingFace model ID for sentence embeddings.
    device:
        Torch device for the embedding model ('cpu', 'cuda', 'cuda:0', …).
    """

    def __init__(
        self,
        *,
        em_weight: float = 0.20,
        rouge_weight: float = 0.25,
        bleu_weight: float = 0.15,
        fuzzy_weight: float = 0.20,
        semantic_weight: float = 0.20,
        threshold: float = 0.60,
        semantic_model_name: str = "BAAI/bge-base-en-v1.5",
        device: str = "cpu",
    ) -> None:
        self.weights = {
            "em": em_weight,
            "rouge": rouge_weight,
            "bleu": bleu_weight,
            "fuzzy": fuzzy_weight,
            "semantic": semantic_weight,
        }
        self.threshold = threshold

        # Normalise weights (in case they don't sum to 1)
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Zero out unavailable signals
        if not _ROUGE_AVAILABLE:
            self.weights["rouge"] = 0.0
        if not _BLEU_AVAILABLE:
            self.weights["bleu"] = 0.0
        if not _FUZZY_AVAILABLE:
            self.weights["fuzzy"] = 0.0
        if not _SEMANTIC_AVAILABLE:
            self.weights["semantic"] = 0.0

        # Re-normalise after zeroing
        total = sum(self.weights.values())
        if total > 0:
            self.weights = {k: v / total for k, v in self.weights.items()}

        # Initialise ROUGE scorer
        self._rouge = None
        if _ROUGE_AVAILABLE and rouge_weight > 0:
            self._rouge = _rouge_scorer_mod.RougeScorer(
                ["rougeL"], use_stemmer=True
            )

        # Initialise sentence-transformer (lazy — loaded on first call)
        self._sem_model = None
        self._sem_model_name = semantic_model_name
        self._device = device
        self._sem_weight = semantic_weight

    # ── Public API ────────────────────────────────────────────────────────────

    def judge(self, predicted: str, reference: str) -> JudgeResult:
        """
        Judge whether *predicted* is semantically equivalent to *reference*.

        Parameters
        ----------
        predicted:
            Model-generated answer string.
        reference:
            Gold-standard answer string.

        Returns
        -------
        JudgeResult with individual signal scores and the final decision.
        """
        pred_norm = _normalise(predicted)
        ref_norm = _normalise(reference)

        em = self._exact_match(pred_norm, ref_norm)
        rouge = self._rouge_l(pred_norm, ref_norm)
        bleu = self._bleu4(pred_norm, ref_norm)
        fuzzy = self._fuzzy(pred_norm, ref_norm)
        sem = self._semantic(predicted, reference)  # raw text for embeddings

        # Weighted sum
        score = (
            self.weights["em"] * (em or 0.0)
            + self.weights["rouge"] * (rouge or 0.0)
            + self.weights["bleu"] * (bleu or 0.0)
            + self.weights["fuzzy"] * (fuzzy or 0.0)
            + self.weights["semantic"] * (sem or 0.0)
        )

        return JudgeResult(
            predicted=predicted,
            reference=reference,
            is_equivalent=score >= self.threshold,
            score=score,
            em_score=em,
            rouge_score=rouge,
            bleu_score=bleu,
            fuzzy_score=fuzzy,
            semantic_score=sem,
            signal_weights=dict(self.weights),
            threshold=self.threshold,
        )

    def judge_batch(
        self,
        predicted_list: List[str],
        reference_list: List[str],
    ) -> List[JudgeResult]:
        """
        Judge a batch of (predicted, reference) pairs.
        Embeddings are computed in one forward pass for efficiency.
        """
        assert len(predicted_list) == len(reference_list)

        # Pre-compute all embeddings at once
        if _SEMANTIC_AVAILABLE and self.weights.get("semantic", 0) > 0:
            model = self._get_sem_model()
            all_texts = predicted_list + reference_list
            all_embs = model.encode(all_texts, convert_to_numpy=True, batch_size=64)
            pred_embs = all_embs[: len(predicted_list)]
            ref_embs = all_embs[len(predicted_list):]
        else:
            pred_embs = ref_embs = [None] * len(predicted_list)

        results = []
        for i, (pred, ref) in enumerate(zip(predicted_list, reference_list)):
            pred_norm = _normalise(pred)
            ref_norm = _normalise(ref)

            em = self._exact_match(pred_norm, ref_norm)
            rouge = self._rouge_l(pred_norm, ref_norm)
            bleu = self._bleu4(pred_norm, ref_norm)
            fuzzy = self._fuzzy(pred_norm, ref_norm)
            sem = (
                _cosine_sim(pred_embs[i], ref_embs[i])
                if pred_embs[i] is not None
                else None
            )

            score = (
                self.weights["em"] * (em or 0.0)
                + self.weights["rouge"] * (rouge or 0.0)
                + self.weights["bleu"] * (bleu or 0.0)
                + self.weights["fuzzy"] * (fuzzy or 0.0)
                + self.weights["semantic"] * (sem or 0.0)
            )

            results.append(
                JudgeResult(
                    predicted=pred,
                    reference=ref,
                    is_equivalent=score >= self.threshold,
                    score=score,
                    em_score=em,
                    rouge_score=rouge,
                    bleu_score=bleu,
                    fuzzy_score=fuzzy,
                    semantic_score=sem,
                    signal_weights=dict(self.weights),
                    threshold=self.threshold,
                )
            )
        return results

    # ── Signal implementations ────────────────────────────────────────────────

    @staticmethod
    def _exact_match(pred: str, ref: str) -> float:
        """Normalised exact match: 1.0 if identical after normalisation."""
        return 1.0 if pred == ref else 0.0

    def _rouge_l(self, pred: str, ref: str) -> Optional[float]:
        if self._rouge is None:
            return None
        try:
            scores = self._rouge.score(ref, pred)
            return scores["rougeL"].fmeasure
        except Exception:
            return None

    @staticmethod
    def _bleu4(pred: str, ref: str) -> Optional[float]:
        if not _BLEU_AVAILABLE:
            return None
        try:
            # sacrebleu sentence_bleu expects hypothesis and list of references
            result = _sacrebleu.sentence_bleu(
                pred, [ref], smooth_method="exp"
            )
            return result.score / 100.0  # convert 0–100 → 0–1
        except Exception:
            return None

    @staticmethod
    def _fuzzy(pred: str, ref: str) -> Optional[float]:
        if not _FUZZY_AVAILABLE:
            return None
        try:
            return _fuzz.token_sort_ratio(pred, ref) / 100.0
        except Exception:
            return None

    def _semantic(self, pred: str, ref: str) -> Optional[float]:
        if not _SEMANTIC_AVAILABLE or self.weights.get("semantic", 0) == 0:
            return None
        try:
            model = self._get_sem_model()
            embs = model.encode([pred, ref], convert_to_numpy=True)
            return float(_cosine_sim(embs[0], embs[1]))
        except Exception as exc:
            logger.warning("Semantic scoring failed: %s", exc)
            return None

    def _get_sem_model(self):
        """Lazy-load the sentence-transformer model."""
        if self._sem_model is None:
            logger.info("Loading semantic model: %s", self._sem_model_name)
            self._sem_model = _SentenceTransformer(
                self._sem_model_name, device=self._device
            )
        return self._sem_model

    @classmethod
    def from_config(cls, cfg: dict) -> "SemanticJudge":
        """Construct from the 'judge' sub-dict of the YAML config."""
        j = cfg.get("judge", {})
        return cls(
            em_weight=j.get("em_weight", 0.20),
            rouge_weight=j.get("rouge_weight", 0.25),
            bleu_weight=j.get("bleu_weight", 0.15),
            fuzzy_weight=j.get("fuzzy_weight", 0.20),
            semantic_weight=j.get("semantic_weight", 0.20),
            threshold=j.get("threshold", 0.60),
            semantic_model_name=j.get("semantic_model", "BAAI/bge-base-en-v1.5"),
        )


# ── Text normalisation ────────────────────────────────────────────────────────

_ARTICLES_RE = re.compile(r"\b(a|an|the)\b", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")


def _normalise(text: str) -> str:
    """
    Normalise answer text for comparison.

    Steps (following the SQuAD evaluation script convention):
      1. Lower-case
      2. Remove punctuation
      3. Remove articles (a, an, the)
      4. Collapse whitespace
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = _ARTICLES_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text).strip()
    return text


# ── Cosine similarity (numpy) ─────────────────────────────────────────────────


def _cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two 1-D numpy vectors."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))
