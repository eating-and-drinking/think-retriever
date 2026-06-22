#!/usr/bin/env python3
"""
train_two_stage.py
==================
Two-Stage Training: RPA + Tree-Search GRPO

Stage 1: RPA (ReAct Protocol Alignment)
   - Learn correct protocol: Think → Search → Content → Answer
   - Standard GRPO with composite rewards
   - Update entire trajectory

Stage 2: Tree-Search GRPO
   - Learn which search is most effective
   - Tree-structured exploration with branching factor G
   - Same-parent grouping → guaranteed contextual fairness
   - Probe mechanism for knowledge state evaluation
   - All tokens (think + search + probe answer) updated
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent / "src"))

from think_retriever.agent.qwen_style_agent import QwenStyleAgent
from think_retriever.judge.probe_evaluator import ProbeEvaluator
from think_retriever.judge.semantic_judge import SemanticJudge
from think_retriever.rewards.two_stage_reward import TwoStageRewardFn
from think_retriever.trainer.two_stage_trainer import (
    QADataset,
    TwoStageConfig,
    TwoStageTrainer,
    apply_lora,
)
from think_retriever.trainer.tree_search_sampler import TreeSearchSampler
from think_retriever.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def load_jsonl(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    logger.info("Loaded %d records from %s", len(records), path)
    return records


class BM25Retriever:
    """Simple BM25-style retriever for demonstration."""
    
    def __init__(self, corpus: list[str] | None = None) -> None:
        self.corpus = corpus or [
            "罗杰·彭罗斯（Roger Penrose）是英国数学家和物理学家，"
            "2020年因对黑洞形成机制的数学证明获得诺贝尔物理学奖。",
            "彭罗斯与霍金合作证明了奇点定理，为广义相对论和黑洞物理奠定了基础。",
            "他还以彭罗斯镶嵌、微管意识理论等闻名。",
            "Penrose, Hawking, and the Nobel Prize in Physics 2020.",
        ]
    
    def search(self, query: str, top_k: int = 3) -> str:
        query_words = set(query.lower().split())
        scored = []
        for doc in self.corpus:
            doc_words = set(doc.lower().split())
            overlap = len(query_words & doc_words)
            if overlap > 0:
                scored.append((overlap, doc))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored:
            return scored[0][1][:800]
        return "No relevant information found."


def main() -> None:
    parser = argparse.ArgumentParser("Two-Stage Trainer: RPA + Tree-Search GRPO")
    parser.add_argument("--config", type=str, default="configs/default.yaml")
    parser.add_argument("--train_path", type=str, default="data/train.jsonl")
    parser.add_argument("--eval_path", type=str, default=None)
    
    # Stage 1 params
    parser.add_argument("--stage1_epochs", type=int, default=None)
    parser.add_argument("--stage1_group_size", type=int, default=None)
    parser.add_argument("--stage1_lr", type=float, default=None)
    
    # Stage 2 params
    parser.add_argument("--stage2_epochs", type=int, default=None)
    parser.add_argument("--stage2_branching_factor", type=int, default=None)
    parser.add_argument("--stage2_max_depth", type=int, default=None)
    parser.add_argument("--stage2_stop_threshold", type=float, default=None)
    parser.add_argument("--stage2_search_cost", type=float, default=None)
    parser.add_argument("--stage2_lr", type=float, default=None)
    
    # Common
    parser.add_argument("--model_name_or_path", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--grad_accum", type=int, default=None)
    parser.add_argument("--log_level", type=str, default="INFO")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--use_lora", action="store_true", help="Train a LoRA adapter instead of full fine-tuning.")
    
    args = parser.parse_args()
    setup_logging(level=args.log_level)
    
    # Load config
    with open(args.config, encoding="utf-8") as f:
        file_cfg = yaml.safe_load(f) or {}
    
    # Build TwoStageConfig
    cfg = TwoStageConfig.from_dict(file_cfg)
    
    # Override with command-line args
    if args.stage1_epochs:
        cfg.stage1_epochs = args.stage1_epochs
    if args.stage1_group_size:
        cfg.stage1_group_size = args.stage1_group_size
    if args.stage1_lr:
        cfg.stage1_lr = args.stage1_lr
    
    if args.stage2_epochs:
        cfg.stage2_epochs = args.stage2_epochs
    if args.stage2_branching_factor:
        cfg.stage2_branching_factor = args.stage2_branching_factor
    if args.stage2_max_depth:
        cfg.stage2_max_depth = args.stage2_max_depth
    if args.stage2_stop_threshold:
        cfg.stage2_stop_threshold = args.stage2_stop_threshold
    if args.stage2_search_cost:
        cfg.stage2_search_cost = args.stage2_search_cost
    if args.stage2_lr:
        cfg.stage2_lr = args.stage2_lr
    
    if args.model_name_or_path:
        cfg.model_name_or_path = args.model_name_or_path
    if args.output_dir:
        cfg.output_dir = args.output_dir
    if args.epochs:
        cfg.stage1_epochs = args.epochs
        cfg.stage2_epochs = max(1, args.epochs // 2)
    if args.grad_accum:
        cfg.gradient_accumulation_steps = args.grad_accum
    if args.use_lora:
        cfg.use_lora = True
    
    logger.info("Two-Stage Config:\n%s", json.dumps(vars(cfg), indent=2, default=str))
    torch.manual_seed(cfg.seed)
    
    # Load data
    train_records = load_jsonl(args.train_path)
    train_ds = QADataset(train_records)
    
    eval_ds = None
    if args.eval_path and Path(args.eval_path).exists():
        eval_records = load_jsonl(args.eval_path)
        eval_ds = QADataset(eval_records)
    
    # Load model & tokenizer
    dtype = torch.float16 if args.fp16 else torch.bfloat16
    logger.info("Loading model: %s (dtype=%s)", cfg.model_name_or_path, dtype)
    
    tokenizer = AutoTokenizer.from_pretrained(cfg.model_name_or_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load the base model; fall back to SDPA attention if flash-attn is unavailable.
    load_kwargs = dict(
        torch_dtype=dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=cfg.attn_implementation,
    )
    try:
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **load_kwargs)
    except (ImportError, ValueError) as exc:
        logger.warning(
            "Failed to load with attn_implementation=%s (%s); retrying with 'sdpa'.",
            cfg.attn_implementation, exc,
        )
        load_kwargs["attn_implementation"] = "sdpa"
        model = AutoModelForCausalLM.from_pretrained(cfg.model_name_or_path, **load_kwargs)

    # Optionally wrap in LoRA (low-memory training; no-op when cfg.use_lora is False).
    model = apply_lora(model, cfg)

    # Full fine-tuning with gradient checkpointing (apply_lora already handles
    # checkpointing for the LoRA path).
    if cfg.gradient_checkpointing and not cfg.use_lora:
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )

    # Initialize components
    retriever = BM25Retriever()
    semantic_judge = SemanticJudge(threshold=cfg.stage2_stop_threshold)
    probe_eval = ProbeEvaluator(judge=semantic_judge, weight=1.0)
    reward_fn = TwoStageRewardFn(judge=semantic_judge)

    # Stage 1 Agent (Qwen style)
    agent = QwenStyleAgent(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        max_new_tokens=cfg.max_completion_length,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
        enable_tools=["search"],
        probe_evaluator=probe_eval,
    )
    
    # Stage 2 Tree Sampler
    tree_sampler = TreeSearchSampler(
        model=model,
        tokenizer=tokenizer,
        retriever=retriever,
        probe_evaluator=probe_eval,
        branching_factor=cfg.stage2_branching_factor,
        max_depth=cfg.stage2_max_depth,
        stop_threshold=cfg.stage2_stop_threshold,
        search_cost=cfg.stage2_search_cost,
        temperature=cfg.temperature,
        top_p=cfg.top_p,
    )
    
    # Create trainer
    trainer = TwoStageTrainer(
        model=model,
        tokenizer=tokenizer,
        agent=agent,
        tree_sampler=tree_sampler,
        reward_fn=reward_fn,
        probe_evaluator=probe_eval,
        config=cfg,
    )
    
    # Start training
    trainer.train(train_dataset=train_ds, eval_dataset=eval_ds)
    logger.info("Two-stage training complete!")


if __name__ == "__main__":
    main()
