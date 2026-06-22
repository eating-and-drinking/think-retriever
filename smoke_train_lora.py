#!/usr/bin/env python3
"""
smoke_train_lora.py
===================
Minimal LoRA smoke-training run for the two-stage (RPA + Tree-Search GRPO)
pipeline, sized to fit on a single GPU with only ~10GB free.

Differences from train_two_stage.py:
  * Wraps the base model in a LoRA adapter (peft) — only adapter params are
    trained, so AdamW state stays tiny and the 3B base stays frozen in bf16.
  * Loads with attn_implementation="sdpa" (flash-attn is not installed).
  * Uses correct constructor signatures (the original entrypoint passed
    kwargs that don't exist on the classes).
  * Tiny settings: small group size, few tokens, 1 epoch — this proves the
    full training loop runs end-to-end and the LoRA weights actually update.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from think_retriever.agent.qwen_style_agent import QwenStyleAgent
from think_retriever.judge.probe_evaluator import ProbeEvaluator
from think_retriever.judge.semantic_judge import SemanticJudge
from think_retriever.rewards.two_stage_reward import TwoStageRewardFn
from think_retriever.trainer.tree_search_sampler import TreeSearchSampler
from think_retriever.trainer.two_stage_trainer import (
    QADataset,
    TwoStageConfig,
    TwoStageTrainer,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("smoke_train")

MODEL_PATH = "/home/yons/file/firedog/Qwen2.5-3B-Instruct"
DATA_PATH = "data/sample/train.jsonl"
OUT_DIR = "outputs/smoke_lora"


class BM25Retriever:
    """Tiny lexical retriever over the sample corpus (good enough for a smoke run)."""

    def __init__(self) -> None:
        self.corpus = [
            "Paris is the capital and most populous city of France.",
            "Berlin is the capital and largest city of Germany.",
            "Tokyo is the capital of Japan and its most populous city.",
            "Rome is the capital city of Italy.",
            "Madrid is the capital of Spain.",
        ]

    def search(self, query: str, top_k: int = 1) -> str:
        q = set(query.lower().split())
        scored = [(len(q & set(d.lower().split())), d) for d in self.corpus]
        scored.sort(key=lambda x: x[0], reverse=True)
        return scored[0][1] if scored and scored[0][0] > 0 else "No relevant information found."


def load_jsonl(path: str):
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def main() -> None:
    torch.manual_seed(0)

    # ── tiny config ──────────────────────────────────────────────────────────
    cfg = TwoStageConfig(
        model_name_or_path=MODEL_PATH,
        stage1_epochs=1,
        stage1_group_size=4,          # 4 rollouts per question → reward variance for GRPO
        stage1_lr=1e-4,               # higher LR for LoRA
        stage2_epochs=1,
        stage2_branching_factor=2,
        stage2_max_depth=1,
        stage2_lr=1e-4,
        gradient_accumulation_steps=1,  # step every sample so we see updates fast
        max_completion_length=256,
        max_prompt_length=512,
        output_dir=OUT_DIR,
        logging_steps=1,
        save_steps=10_000,            # don't checkpoint during the smoke run
        eval_steps=10_000,
    )

    logger.info("Loading tokenizer + base model (bf16, sdpa) from %s", MODEL_PATH)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",   # flash-attn not installed
        trust_remote_code=True,
    ).to("cuda")

    # ── wrap in LoRA: only adapter params train, base stays frozen ───────────
    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.0,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    model.config.use_cache = False
    # Gradient checkpointing keeps activation memory low enough for ~10GB free.
    # use_reentrant=False is required so grads still flow to the LoRA params
    # when the base model is frozen (reentrant mode needs an input requiring grad).
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.enable_input_require_grads()

    # shorten the agent's generation budget to keep memory + time small
    agent_max_new_tokens = 192
    agent_max_calls = 2

    retriever = BM25Retriever()
    judge = SemanticJudge(threshold=cfg.stage2_stop_threshold, device="cpu")
    probe_eval = ProbeEvaluator(judge=judge, weight=1.0)
    reward_fn = TwoStageRewardFn(judge=judge)

    agent = QwenStyleAgent(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        max_calls=agent_max_calls,
        max_new_tokens=agent_max_new_tokens,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        enable_tools=["search"],
        probe_evaluator=probe_eval,
    )

    tree_sampler = TreeSearchSampler(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        probe_evaluator=probe_eval,
        branching_factor=cfg.stage2_branching_factor,
        max_depth=cfg.stage2_max_depth,
        stop_threshold=cfg.stage2_stop_threshold,
        search_cost=cfg.stage2_search_cost,
        max_search_tokens=128,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )

    trainer = TwoStageTrainer(
        model=model,
        tokenizer=tokenizer,
        agent=agent,
        tree_sampler=tree_sampler,
        reward_fn=reward_fn,
        probe_evaluator=probe_eval,
        config=cfg,
    )

    # rebuild the optimizer over ONLY the trainable (LoRA) params
    trainable = [p for p in model.parameters() if p.requires_grad]
    trainer.optimizer = torch.optim.AdamW(trainable, lr=cfg.stage1_lr)
    logger.info("Optimizer rebuilt over %d trainable tensors", len(trainable))

    # snapshot a LoRA weight to confirm it actually changes after training
    probe_name = next(n for n, p in model.named_parameters() if p.requires_grad and "lora_A" in n)
    before = model.state_dict()[probe_name].detach().float().clone()

    records = load_jsonl(DATA_PATH)[:3]   # only a few questions for the smoke run
    train_ds = QADataset(records)
    logger.info("Smoke training on %d questions", len(train_ds))

    trainer.train(train_dataset=train_ds, eval_dataset=None)

    after = model.state_dict()[probe_name].detach().float()
    delta = (after - before).abs().mean().item()
    logger.info("LoRA weight '%s' mean |Δ| after training = %.3e", probe_name, delta)
    logger.info("GPU peak mem: %.2f GB", torch.cuda.max_memory_allocated() / 1e9)
    print(f"\nSMOKE_RESULT weight_delta={delta:.3e} peak_gb={torch.cuda.max_memory_allocated()/1e9:.2f}")


if __name__ == "__main__":
    main()
