"""
tree_trainer.py
===============
Tree-Structured Search GRPO Trainer

核心思想（用户设计）：
  1. 用树的方式生成轨迹：Question -> [G个Think+Search] -> [G个ProbeAnswer]
  2. 只有共享相同父节点的节点才在同一组内做优势归一化
     -> 保证组内所有节点有完全相同的上下文
  3. 组内优势 = (r_i - μ_group) / (σ_group + ε)
  4. 所有模型生成的 token（think + search + probe answer）都参与损失计算
  5. 终端节点直接用最终答案评分，不需要单独的 "answer reward"

每个节点的损失：
  L_node = -Advantage * [log π_θ(think_tokens) + log π_θ(search_tokens) + log π_θ(answer_tokens)]

整棵树的损失是所有节点损失的平均。
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from transformers import (
    AutoModelForCausalLM,
    PreTrainedModel,
    PreTrainedTokenizerBase,
)

from think_retriever.judge.probe_evaluator import ProbeEvaluator
from think_retriever.trainer.tree_search_sampler import (
    SearchNode,
    TreeEpisode,
    TreeSearchSampler,
)

logger = logging.getLogger(__name__)


# ── Config ────────────────────────────────────────────────────────────────────

@dataclass
class TreeTrainerConfig:
    """Tree-Search GRPO 训练配置。"""
    
    # 模型
    model_name_or_path: str = "Qwen/Qwen2.5-7B-Instruct"
    
    # 树搜索参数
    branching_factor: int = 4           # G: 每个父节点展开的子节点数
    max_depth: int = 3                   # 最大搜索深度
    stop_threshold: float = 0.9          # tau: ProbeScore >= tau 时停止扩展
    search_cost: float = 0.05            # lambda: 每次搜索的成本
    max_search_tokens: int = 128         # think + search 最多生成多少 token
    
    # 训练参数
    learning_rate: float = 5e-7
    gradient_accumulation_steps: int = 8
    max_grad_norm: float = 1.0
    clip_range: float = 0.2              # PPO-style clipping (optional)
    epsilon: float = 1e-8
    temperature: float = 1.0
    top_p: float = 0.95
    num_train_epochs: int = 2
    batch_size: int = 1                  # 每个 step 一个 question 的树
    
    # 计算
    use_clipping: bool = False            # 是否用 PPO clipping
    
    # 输出 & 日志
    output_dir: str = "outputs/tree_search/"
    logging_steps: int = 10
    save_steps: int = 100
    save_total_limit: int = 3
    run_name: str = "tree-search-grpo"
    report_to: str = "none"
    seed: int = 42


# ── Dataset ───────────────────────────────────────────────────────────────────

class QADataset(Dataset):
    """简单的 QA 数据集。"""
    
    def __init__(self, records: List[dict]) -> None:
        self.records = records
    
    def __len__(self) -> int:
        return len(self.records)
    
    def __getitem__(self, idx: int) -> dict:
        r = self.records[idx]
        return {
            "question": r.get("question", str(r)),
            "answer": r.get("answer", ""),
        }


# ── Loss Computation ───────────────────────────────────────────────────────────

def compute_tree_loss(
    episode: TreeEpisode,
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizerBase,
    config: TreeTrainerConfig,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    计算一整棵树的总损失。
    
    奖励差异化设计：
    - 最终组（is_final_group=True）：计算 think + search + answer 的损失
    - 其他节点：只计算 think + search 的损失
    - 目的：防止模型在非最终节点提前回答
    
    对每个 SearchNode：
    1. 检查组内归一化的 advantage（已在采样时计算好）
    2. 对该节点所有"模型生成的 token"（think + search [+ probe answer]）
       计算对数概率
    3. loss = -advantage * sum(log_prob)
    
    Parameters
    ----------
    episode: TreeEpisode
        一棵树的所有节点和组信息
    model: 当前训练的模型
    tokenizer
    config
    
    Returns
    -------
    total_loss: scalar tensor
    metrics: 各类统计指标
    """
    
    total_loss = torch.tensor(0.0, device=model.device)
    n_nodes_with_loss = 0
    
    # 统计信息
    total_advantage_abs = 0.0
    n_pos_adv = 0
    n_neg_adv = 0
    n_zero_adv = 0
    total_log_prob = 0.0
    total_tokens = 0
    n_answer_tokens = 0
    n_nodes_with_answer = 0
    
    for group in episode.groups:
        is_final = group.is_final_group  # 是否为最终组
        
        for node in group.nodes:
            adv = node.advantage
            
            # 统计
            if adv > 0:
                n_pos_adv += 1
            elif adv < 0:
                n_neg_adv += 1
            else:
                n_zero_adv += 1
            total_advantage_abs += abs(adv)
            
            # advantage 为 0 时不需要计算梯度
            if abs(adv) < 1e-6:
                continue
            
            # ========== 计算此节点的 log probability ==========
            
            # ---- A. Think + Search tokens （从搜索采样时已保存 input_ids）----
            if node.input_ids is not None and node.generation_start_pos > 0:
                # .clone() converts the inference-mode tensor (produced during
                # sampling) into a normal tensor that autograd can save for backward.
                input_ids = node.input_ids.to(model.device).clone()
                gen_start = node.generation_start_pos
                
                with torch.enable_grad():
                    logits = model(input_ids=input_ids).logits
                
                # 对 [gen_start, end] 这一段生成的token计算log_prob
                log_probs = F.log_softmax(logits[0, gen_start:-1, :], dim=-1)
                target_ids = input_ids[0, gen_start + 1:]
                
                gathered = log_probs.gather(
                    dim=-1,
                    index=target_ids.unsqueeze(-1)
                ).squeeze(-1)
                
                # 乘以 advantage（注意符号：正 advantage -> 增加概率 -> -A * log_p 当 A>0 时为负）
                node_loss = -float(adv) * gathered.sum()
                
                total_loss = total_loss + node_loss
                n_nodes_with_loss += 1
                total_log_prob += gathered.sum().item()
                total_tokens += gathered.numel()
            
            # ---- B. Probe Answer tokens（仅最终组计算）----
            # 只有最终组才计算 answer 的损失，防止模型在非最终节点提前回答
            if is_final and node.answer_token_range is not None and node.probe_answer:
                ans_start, ans_end = node.answer_token_range
                
                # 构造一个 prompt + answer 的输入（用原始 sampler 的 prompt）
                # 为了节省重新推理的开销，这里只做增量计算
                # 具体做法：把 answer 的 token 拿出来，重新在同一 prompt 下做 teacher forcing log_prob
                
                probe_prompt = _rebuild_probe_prompt(node)
                if probe_prompt:
                    full_text = probe_prompt + node.probe_answer
                    probe_input_ids = tokenizer(
                        full_text, return_tensors="pt", truncation=True, max_length=2048
                    ).input_ids.to(model.device)
                    
                    prompt_len = tokenizer(
                        probe_prompt, return_tensors="pt", truncation=True, max_length=2048
                    ).input_ids.shape[1]
                    
                    with torch.enable_grad():
                        probe_logits = model(input_ids=probe_input_ids).logits
                    
                    # answer 部分的 log_prob
                    answer_log_probs = F.log_softmax(
                        probe_logits[0, prompt_len:-1, :], dim=-1
                    )
                    answer_target = probe_input_ids[0, prompt_len + 1:]
                    
                    answer_gathered = answer_log_probs.gather(
                        dim=-1,
                        index=answer_target.unsqueeze(-1)
                    ).squeeze(-1)
                    
                    answer_loss = -float(adv) * answer_gathered.sum()
                    total_loss = total_loss + answer_loss
                    total_log_prob += answer_gathered.sum().item()
                    total_tokens += answer_gathered.numel()
                    n_answer_tokens += answer_gathered.numel()
                    n_nodes_with_answer += 1
    
    # 平均损失（按参与计算的节点数）
    if n_nodes_with_loss > 0:
        avg_loss = total_loss / n_nodes_with_loss
    else:
        # 如果所有节点 advantage 都为 0（极端情况），返回 0 loss
        avg_loss = torch.tensor(0.0, device=model.device, requires_grad=True)
    
    metrics = {
        "loss/tree_avg_loss": avg_loss.item() if avg_loss.numel() > 0 else 0.0,
        "loss/total_nodes": len(episode.all_nodes),
        "loss/nodes_with_loss": n_nodes_with_loss,
        "loss/n_pos_advantage": n_pos_adv,
        "loss/n_neg_advantage": n_neg_adv,
        "loss/n_zero_advantage": n_zero_adv,
        "loss/avg_abs_advantage": total_advantage_abs / max(len(episode.all_nodes), 1),
        "loss/avg_log_prob_per_token": total_log_prob / max(total_tokens, 1),
        "tree/total_groups": len(episode.groups),
        "tree/max_depth": episode.max_reached_depth(),
        "tree/nodes_with_answer": n_nodes_with_answer,
        "tree/answer_token_ratio": n_answer_tokens / max(total_tokens, 1),
    }
    
    return avg_loss, metrics


def _rebuild_probe_prompt(node: SearchNode) -> Optional[str]:
    """从 SearchNode 信息重建 probe prompt（用于 answer token 的 log_prob 计算）。"""
    if not node.content_text or not node.question:
        return None
    
    # 注意：这里需要与 sampler 里 _build_probe_prompt 的格式保持一致
    # 简化版做法：只基于 content_text 生成一个简单的 QA prompt
    return (
        f"<|im_start|>system\n"
        f"Answer the question based on the search result.\n"
        f"<|im_end|>\n"
        f"<|im_start|>user\n"
        f"Question: {node.question}\n"
        f"Search result: {node.content_text[:500]}\n"
        f"<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )


# ── Trainer ───────────────────────────────────────────────────────────────────

class TreeTrainer:
    """
    Tree-Search GRPO 主训练器。
    
    流程：
    1. 采样：对每个 question，用 TreeSearchSampler 生成一棵树状轨迹
    2. 评估：每个节点的 ProbeScore 已在采样时计算好
    3. 优势：每组内（同父节点）做归一化，得到每个节点的 advantage
    4. 损失：-advantage * sum(log_prob(所有生成token))
    5. 更新：optimizer.step()
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        sampler: TreeSearchSampler,
        config: TreeTrainerConfig,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.sampler = sampler
        self.cfg = config
        
        # 优化器
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
            weight_decay=0.0,
        )
        
        # wandb（可选）
        self._wandb = None
        if config.report_to == "wandb":
            try:
                import wandb
                wandb.init(project="tree-search-grpo", name=config.run_name)
                self._wandb = wandb
            except ImportError:
                logger.warning("wandb not installed, skipping")
    
    # ── Public API ──────────────────────────────────────────────────────────
    
    def train(
        self,
        dataset: QADataset,
        eval_dataset: Optional[QADataset] = None,
    ) -> None:
        """主训练循环。"""
        torch.manual_seed(self.cfg.seed)
        
        loader = DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=0,
            collate_fn=lambda batch: batch,
        )
        
        global_step = 0
        accum_count = 0
        
        logger.info(
            "Starting training: epochs=%d, batch_size=%d, G=%d, max_depth=%d",
            self.cfg.num_train_epochs, self.cfg.batch_size,
            self.cfg.branching_factor, self.cfg.max_depth,
        )
        
        for epoch in range(self.cfg.num_train_epochs):
            logger.info("=== Epoch %d / %d ===", epoch + 1, self.cfg.num_train_epochs)
            
            for batch in loader:
                # 每个 item 是一个 (question, answer)
                for item in batch:
                    question = item["question"]
                    reference = item["answer"]
                    
                    # Step 1: 采样整棵树
                    with torch.inference_mode():
                        episode = self.sampler.sample_tree(question, reference)
                    
                    # Step 2: 计算树损失（带梯度）
                    loss, metrics = compute_tree_loss(episode, self.model, self.tokenizer, self.cfg)
                    
                    # Step 3: 反向传播 + 梯度累积
                    (loss / self.cfg.gradient_accumulation_steps).backward()
                    accum_count += 1
                    
                    metrics["episode/nodes"] = episode.total_nodes()
                    metrics["episode/groups"] = len(episode.groups)
                    metrics["episode/terminal_nodes"] = len(episode.terminal_nodes)
                    if episode.terminal_nodes:
                        best_q = max((n.probe_score for n in episode.terminal_nodes), default=0.0)
                        metrics["eval/best_terminal_probe_score"] = best_q
                    
                    if global_step % self.cfg.logging_steps == 0:
                        self._log(metrics, global_step)
                    
                    # Step 4: 梯度更新
                    if accum_count >= self.cfg.gradient_accumulation_steps:
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(), self.cfg.max_grad_norm
                        )
                        self.optimizer.step()
                        self.optimizer.zero_grad()
                        accum_count = 0
                        global_step += 1
                        
                        logger.info(
                            "step %d: loss=%.4f, nodes=%d, groups=%d, best_Q=%.3f",
                            global_step, metrics["loss/tree_avg_loss"],
                            metrics["episode/nodes"], metrics["episode/groups"],
                            metrics.get("eval/best_terminal_probe_score", 0.0),
                        )
                        
                        if global_step % self.cfg.save_steps == 0:
                            self._save(f"checkpoint-{global_step}")
                        
                        if eval_dataset and global_step % 50 == 0:
                            eval_metrics = self.evaluate(eval_dataset, num_samples=8)
                            logger.info("Eval @ step %d: %s", global_step, eval_metrics)
                            self._log({f"eval/{k}": v for k, v in eval_metrics.items()}, global_step)
            
            logger.info("Epoch %d done", epoch + 1)
            self._save(f"checkpoint-epoch-{epoch + 1}")
        
        self._save("final")
        logger.info("Training complete!")
    
    # ── Evaluation ─────────────────────────────────────────────────────────
    
    @torch.inference_mode()
    def evaluate(
        self,
        dataset: QADataset,
        num_samples: int = 16,
    ) -> Dict[str, float]:
        """简单的评估：采样几棵树，看终端节点的最高 ProbeScore。"""
        indices = list(range(min(num_samples, len(dataset))))
        
        all_probe_scores: List[float] = []
        all_search_counts: List[int] = []
        all_depths: List[int] = []
        
        for idx in indices:
            item = dataset[idx]
            ep = self.sampler.sample_tree(item["question"], item["answer"])
            
            if ep.terminal_nodes:
                best = max((n.probe_score for n in ep.terminal_nodes), default=0.0)
                all_probe_scores.append(best)
            
            all_search_counts.append(ep.total_nodes())
            all_depths.append(ep.max_reached_depth())
        
        return {
            "avg_best_probe_score": sum(all_probe_scores) / max(len(all_probe_scores), 1),
            "avg_search_count": sum(all_search_counts) / max(len(all_search_counts), 1),
            "avg_max_depth": sum(all_depths) / max(len(all_depths), 1),
            "n_questions": len(all_search_counts),
        }
    
    # ── Helpers ────────────────────────────────────────────────────────────
    
    def _save(self, tag: str) -> None:
        import os
        path = f"{self.cfg.output_dir}/{tag}"
        os.makedirs(path, exist_ok=True)
        
        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        
        # 保存配置
        config_path = f"{path}/tree_trainer_config.json"
        with open(config_path, "w") as f:
            json.dump(vars(self.cfg), f, indent=2)
        
        logger.info("Saved checkpoint to %s", path)
    
    def _log(self, metrics: Dict[str, float], step: int) -> None:
        if self._wandb:
            try:
                self._wandb.log(metrics, step=step)
            except Exception:
                pass
