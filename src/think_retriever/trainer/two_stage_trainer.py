"""
two_stage_trainer.py
====================
Two-Stage Trainer for RPA + Tree-Search GRPO

Stage 1: RPA (ReAct Protocol Alignment)
    - Learn correct protocol flow (Think → Search → Content → Answer)
    - Standard GRPO with composite rewards (format + protocol + budget + answer)
    - Update entire trajectory

Stage 2: Tree-Search GRPO
    - Learn which search is most effective
    - Tree-structured exploration with branching factor G
    - Same-parent grouping → guaranteed contextual fairness
    - Probe mechanism for knowledge state evaluation
    - All tokens (think + search + probe answer) updated

Key improvements from v1:
  - Stage 2 now uses TreeSearchSampler instead of linear exploration
  - Groups are naturally formed by shared parent context
  - All generation tokens participate in loss calculation
  - No wasted samples (group size = branching_factor)
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    get_cosine_schedule_with_warmup,
)

from think_retriever.agent.qwen_style_agent import Episode, QwenStyleAgent
from think_retriever.judge.probe_evaluator import ProbeEvaluator
from think_retriever.judge.semantic_judge import SemanticJudge
from think_retriever.rewards.two_stage_reward import TwoStageRewardFn, extract_final_answer
from think_retriever.trainer.tree_search_sampler import TreeEpisode, TreeSearchSampler
from think_retriever.trainer.tree_trainer import compute_tree_loss

logger = logging.getLogger(__name__)


# ── Config ─────────────────────────────────────────────────────────────────────


@dataclass
class TwoStageConfig:
    """Configuration for two-stage training."""

    # Model
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    torch_dtype: str = "bfloat16"
    attn_implementation: str = "flash_attention_2"

    # Stage 1: RPA (ReAct Protocol Alignment)
    stage1_epochs: int = 2
    stage1_group_size: int = 8
    stage1_lr: float = 5e-7

    # Stage 2: Tree-Search GRPO
    stage2_epochs: int = 1
    stage2_branching_factor: int = 4  # G: children per parent
    stage2_max_depth: int = 3         # Max search depth
    stage2_stop_threshold: float = 0.9  # τ: stop when Q >= τ
    stage2_search_cost: float = 0.05    # λ: cost per search
    stage2_lr: float = 1e-7

    # Common
    per_device_batch_size: int = 1
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    clip_range: float = 0.2
    temperature: float = 1.0
    top_p: float = 0.95
    max_prompt_length: int = 512
    max_completion_length: int = 1024
    warmup_ratio: float = 0.05

    # Training
    output_dir: str = "outputs/two_stage/"
    logging_steps: int = 10
    eval_steps: int = 50
    save_steps: int = 100
    save_total_limit: int = 3
    seed: int = 42
    report_to: str = "none"
    run_name: str = "two-stage-rl"

    @classmethod
    def from_dict(cls, d: dict) -> "TwoStageConfig":
        """Create from config dict."""
        model = d.get("model", {})
        training = d.get("training", {})
        stage1 = d.get("stage1", {})
        stage2 = d.get("stage2", {})

        return cls(
            model_name_or_path=model.get("name_or_path", cls.model_name_or_path),
            torch_dtype=model.get("torch_dtype", cls.torch_dtype),
            attn_implementation=model.get("attn_implementation", cls.attn_implementation),
            stage1_epochs=stage1.get("epochs", cls.stage1_epochs),
            stage1_group_size=stage1.get("group_size", cls.stage1_group_size),
            stage1_lr=stage1.get("lr", cls.stage1_lr),
            stage2_epochs=stage2.get("epochs", cls.stage2_epochs),
            stage2_branching_factor=stage2.get("branching_factor", cls.stage2_branching_factor),
            stage2_max_depth=stage2.get("max_depth", cls.stage2_max_depth),
            stage2_stop_threshold=stage2.get("stop_threshold", cls.stage2_stop_threshold),
            stage2_search_cost=stage2.get("search_cost", cls.stage2_search_cost),
            stage2_lr=stage2.get("lr", cls.stage2_lr),
            per_device_batch_size=training.get("batch_size", cls.per_device_batch_size),
            gradient_accumulation_steps=training.get("grad_accum", cls.gradient_accumulation_steps),
            max_grad_norm=training.get("max_grad_norm", cls.max_grad_norm),
            clip_range=training.get("clip_range", cls.clip_range),
            temperature=training.get("temperature", cls.temperature),
            top_p=training.get("top_p", cls.top_p),
            max_prompt_length=d.get("data", {}).get("max_prompt_length", cls.max_prompt_length),
            max_completion_length=d.get("data", {}).get("max_completion_length", cls.max_completion_length),
            warmup_ratio=training.get("warmup_ratio", cls.warmup_ratio),
            output_dir=training.get("output_dir", cls.output_dir),
            logging_steps=training.get("logging_steps", cls.logging_steps),
            eval_steps=training.get("eval_steps", cls.eval_steps),
            save_steps=training.get("save_steps", cls.save_steps),
            save_total_limit=training.get("save_total_limit", cls.save_total_limit),
            seed=training.get("seed", cls.seed),
            report_to=training.get("report_to", cls.report_to),
            run_name=training.get("run_name", cls.run_name),
        )


# ── Dataset ───────────────────────────────────────────────────────────────────


class QADataset(Dataset):
    """Simple (question, answer) dataset."""

    def __init__(self, records: List[dict], question_field: str = "question", answer_field: str = "answer") -> None:
        self.records = records
        self.q_field = question_field
        self.a_field = answer_field

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {"question": r[self.q_field], "answer": r[self.a_field]}


# ── Stage 1: RPA Trainer ──────────────────────────────────────────────────────


def compute_rpa_loss(
    episodes: List[Episode],
    rewards: List[float],
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: TwoStageConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute Stage 1 (RPA) loss using standard GRPO.
    
    Loss: -A_i * sum(log π_θ(tokens))
    where A_i = (r_i - μ_group) / (σ_group + ε)
    """
    eps = 1e-8
    
    # Normalize rewards within group
    r_t = torch.tensor(rewards, dtype=torch.float32)
    mu = r_t.mean()
    sigma = r_t.std()
    
    if sigma < eps:
        sigma = torch.tensor(1.0)
    
    advantages = ((r_t - mu) / (sigma + eps)).tolist()
    
    total_loss = torch.tensor(0.0, device=model.device)
    n_samples = 0
    
    for episode, adv in zip(episodes, advantages):
        if abs(adv) < 1e-6:
            continue
        
        prompt = episode.completion[:100]  # Simplified: use completion directly
        full_text = prompt + episode.completion
        
        enc = tokenizer(
            full_text,
            return_tensors="pt",
            truncation=True,
            max_length=config.max_prompt_length + config.max_completion_length,
        ).to(model.device)
        
        input_ids = enc.input_ids
        
        with torch.enable_grad():
            logits = model(input_ids=input_ids).logits
        
        log_probs = F.log_softmax(logits[0, :-1, :], dim=-1)
        targets = input_ids[0, 1:]
        
        gathered = log_probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
        loss = -float(adv) * gathered.sum()
        
        total_loss = total_loss + loss
        n_samples += 1
    
    avg_loss = total_loss / max(n_samples, 1)
    
    return avg_loss, {
        "stage1/loss": avg_loss.item(),
        "stage1/num_samples": n_samples,
        "stage1/mu_reward": mu.item(),
        "stage1/sigma_reward": sigma.item(),
    }


# ── Two-Stage Trainer ─────────────────────────────────────────────────────────


class TwoStageTrainer:
    """
    Two-Stage Trainer:
    - Stage 1: RPA (ReAct Protocol Alignment)
    - Stage 2: Tree-Search GRPO
    """

    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        agent: QwenStyleAgent,
        tree_sampler: TreeSearchSampler,
        reward_fn: TwoStageRewardFn,
        probe_evaluator: ProbeEvaluator,
        config: TwoStageConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.agent = agent
        self.tree_sampler = tree_sampler
        self.reward_fn = reward_fn
        self.probe_evaluator = probe_evaluator
        self.cfg = config
        self._global_step = 0

        # Optimizer (shared across stages, LR adjusted per stage)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.stage1_lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )

        Path(config.output_dir).mkdir(parents=True, exist_ok=True)
        self._wandb = self._init_wandb()

    def _init_wandb(self):
        if self.cfg.report_to == "wandb":
            try:
                import wandb
                wandb.init(project="two-stage-rl", name=self.cfg.run_name)
                return wandb
            except ImportError:
                logger.warning("wandb not installed; skipping.")
        return None

    def train(
        self,
        train_dataset: QADataset,
        eval_dataset: Optional[QADataset] = None,
    ) -> None:
        """Main training loop."""
        torch.manual_seed(self.cfg.seed)

        logger.info("=" * 60)
        logger.info("Starting Two-Stage Training")
        logger.info("=" * 60)

        # ── Stage 1: RPA ────────────────────────────────────────────────────────
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 1: RPA (ReAct Protocol Alignment)")
        logger.info("=" * 60)

        self._set_lr(self.cfg.stage1_lr)
        self._train_stage_rpa(train_dataset, eval_dataset, self.cfg.stage1_epochs)

        # ── Stage 2: Tree-Search GRPO ──────────────────────────────────────────
        logger.info("")
        logger.info("=" * 60)
        logger.info("STAGE 2: Tree-Search GRPO")
        logger.info("=" * 60)

        self._set_lr(self.cfg.stage2_lr)
        self._train_stage_tree_search(train_dataset, eval_dataset, self.cfg.stage2_epochs)

        logger.info("=" * 60)
        logger.info("Training Complete!")
        logger.info("=" * 60)

        self._save("final")

    def _set_lr(self, lr: float) -> None:
        """Set learning rate for optimizer."""
        for pg in self.optimizer.param_groups:
            pg["lr"] = lr
        logger.info("Set learning rate to %.2e", lr)

    def _train_stage_rpa(
        self,
        dataset: QADataset,
        eval_dataset: Optional[QADataset],
        num_epochs: int,
    ) -> None:
        """Train Stage 1: RPA."""
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.per_device_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_fn,
        )

        self.model.train()
        accum = 0

        for epoch in range(num_epochs):
            logger.info("=== Stage 1 Epoch %d/%d ===", epoch + 1, num_epochs)

            for batch in loader:
                questions = batch["questions"]
                answers = batch["answers"]

                # Rollout: generate G episodes per question
                all_episodes: List[List[Episode]] = []
                for q in questions:
                    episodes = self.agent.rollout_group(
                        question=q,
                        ground_truth=answers[0] if answers else None,
                        group_size=self.cfg.stage1_group_size,
                        enable_probe=False,
                        temperature=self.cfg.temperature,
                        top_p=self.cfg.top_p,
                    )
                    all_episodes.append(episodes)

                # Compute rewards for each question's group
                for i, episodes in enumerate(all_episodes):
                    q = questions[i]
                    ref = answers[i] if i < len(answers) else ""

                    rewards = []
                    for ep in episodes:
                        # Stage 1: Use RPA rewards (format + protocol + budget + answer)
                        pred_answer = extract_final_answer(ep.completion)
                        reward = self._compute_rpa_reward(
                            completion=ep.completion,
                            reference_answer=ref,
                            num_searches=ep.num_searches,
                        )
                        rewards.append(reward)

                    # Compute loss
                    loss, metrics = compute_rpa_loss(
                        episodes=episodes,
                        rewards=rewards,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        config=self.cfg,
                    )

                    (loss / self.cfg.gradient_accumulation_steps).backward()
                    accum += 1

                    if accum % self.cfg.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self._global_step += 1

                        if self._global_step % self.cfg.logging_steps == 0:
                            self._log(metrics)

                        if eval_dataset and self._global_step % self.cfg.eval_steps == 0:
                            eval_metrics = self._evaluate_stage1(eval_dataset)
                            self._log({f"eval/{k}": v for k, v in eval_metrics.items()})

            self._save(f"stage1-epoch-{epoch + 1}")

    def _compute_rpa_reward(
        self,
        completion: str,
        reference_answer: str,
        num_searches: int,
    ) -> float:
        """Compute Stage 1 RPA reward."""
        # Format reward
        format_ok = "<think>" in completion and "<tool_call>" in completion
        r_format = 1.0 if format_ok else -1.0

        # Protocol reward (think before search)
        think_pos = completion.find("<think>")
        search_pos = completion.find("<tool_call>")
        r_protocol = 1.0 if think_pos >= 0 and search_pos > think_pos else 0.5

        # Answer thinking reward (think before final answer)
        r_answer_thinking = self._compute_answer_thinking_reward(completion)

        # Budget reward
        r_budget = 0.0 if num_searches <= 3 else -0.2 * (num_searches - 3)

        # Answer reward
        pred_answer = extract_final_answer(completion)
        if pred_answer and reference_answer:
            score = self.probe_evaluator.evaluate(pred_answer, reference_answer)
            r_answer = score
        else:
            r_answer = 0.0

        # Total (weighted)
        total = 0.2 * r_format + 0.2 * r_protocol + 0.1 * r_budget + 0.4 * r_answer + 0.1 * r_answer_thinking
        return total

    def _compute_answer_thinking_reward(self, completion: str) -> float:
        """
        Compute reward for having a <think> block immediately before the <answer> block.

        Returns +1.0 if there is a think block before <answer>, -0.5 otherwise.
        Returns 0.0 if no <answer> tag found (partial output).
        """
        import re

        # Find the <answer> open tag position
        answer_match = re.search(r"<answer>", completion)
        if not answer_match:
            return 0.0

        answer_start = answer_match.start()

        # Find all </think> close tag positions
        think_closes = list(re.finditer(r"</think>", completion))
        if not think_closes:
            return -0.5

        # Check that the last </think> before <answer> is close to it
        for think_close in reversed(think_closes):
            if think_close.end() <= answer_start:
                gap_text = completion[think_close.end():answer_start].strip()
                # Allow some whitespace/newlines between </think> and <answer>
                if len(gap_text) < 20:
                    return 1.0
                break

        return -0.5

    def _train_stage_tree_search(
        self,
        dataset: QADataset,
        eval_dataset: Optional[QADataset],
        num_epochs: int,
    ) -> None:
        """Train Stage 2: Tree-Search GRPO."""
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.per_device_batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=self._collate_fn,
        )

        self.model.train()
        accum = 0

        for epoch in range(num_epochs):
            logger.info("=== Stage 2 Epoch %d/%d ===", epoch + 1, num_epochs)

            for batch in loader:
                questions = batch["questions"]
                answers = batch["answers"]

                for i, q in enumerate(questions):
                    ref = answers[i] if i < len(answers) else ""

                    # Sample tree for this question
                    with torch.inference_mode():
                        episode = self.tree_sampler.sample_tree(q, ref)

                    # Compute tree loss
                    loss, metrics = compute_tree_loss(
                        episode=episode,
                        model=self.model,
                        tokenizer=self.tokenizer,
                        config=self.cfg,
                    )

                    (loss / self.cfg.gradient_accumulation_steps).backward()
                    accum += 1

                    if accum % self.cfg.gradient_accumulation_steps == 0:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        self._global_step += 1

                        metrics["stage"] = 2
                        if self._global_step % self.cfg.logging_steps == 0:
                            self._log(metrics)

                        if eval_dataset and self._global_step % self.cfg.eval_steps == 0:
                            eval_metrics = self._evaluate_stage2(eval_dataset)
                            self._log({f"eval/{k}": v for k, v in eval_metrics.items()})

            self._save(f"stage2-epoch-{epoch + 1}")

    def _evaluate_stage1(self, dataset: QADataset, num_samples: int = 32) -> Dict[str, float]:
        """Evaluate Stage 1: Check format compliance and answer accuracy."""
        self.model.eval()
        indices = list(range(min(num_samples, len(dataset))))

        format_ok = 0
        correct = 0
        total_searches = 0

        for idx in indices:
            item = dataset[idx]
            ep = self.agent.rollout(
                item["question"],
                ground_truth=item["answer"],
                enable_probe=False,
                temperature=0.0,
            )

            if "<think>" in ep.completion and "<tool_call>" in ep.completion:
                format_ok += 1

            pred_answer = extract_final_answer(ep.completion)
            if pred_answer:
                score = self.probe_evaluator.evaluate(pred_answer, item["answer"])
                if score >= 0.7:
                    correct += 1

            total_searches += ep.num_searches

        self.model.train()
        n = len(indices)
        return {
            "stage1/format_pass_rate": format_ok / n,
            "stage1/accuracy": correct / n,
            "stage1/avg_searches": total_searches / n,
        }

    def _evaluate_stage2(self, dataset: QADataset, num_samples: int = 16) -> Dict[str, float]:
        """Evaluate Stage 2: Check tree quality and final answer quality."""
        self.model.eval()
        indices = list(range(min(num_samples, len(dataset))))

        best_scores = []
        total_nodes = []
        max_depths = []

        for idx in indices:
            item = dataset[idx]
            ep = self.tree_sampler.sample_tree(item["question"], item["answer"])

            if ep.terminal_nodes:
                best = max((n.probe_score for n in ep.terminal_nodes), default=0.0)
                best_scores.append(best)

            total_nodes.append(ep.total_nodes())
            max_depths.append(ep.max_reached_depth())

        self.model.train()
        n = len(indices)
        return {
            "stage2/best_probe_score": sum(best_scores) / max(len(best_scores), 1),
            "stage2/avg_nodes": sum(total_nodes) / n,
            "stage2/avg_depth": sum(max_depths) / n,
        }

    def _collate_fn(self, batch: List[dict]) -> dict:
        return {
            "questions": [x["question"] for x in batch],
            "answers": [x["answer"] for x in batch],
        }

    def _save(self, tag: str) -> None:
        path = Path(self.cfg.output_dir) / tag
        path.mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        logger.info("Saved checkpoint to %s", path)

    def _log(self, metrics: Dict) -> None:
        msg = " | ".join(
            f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
            for k, v in metrics.items()
        )
        logger.info("[step %d] %s", self._global_step, msg)
        if self._wandb:
            try:
                self._wandb.log(metrics, step=self._global_step)
            except Exception:
                pass
