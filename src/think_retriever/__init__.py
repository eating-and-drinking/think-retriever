# src/think_retriever/__init__.py
"""Two-Stage RL Training for Think-Retriever."""

from .agent.qwen_style_agent import QwenStyleAgent, Episode, ToolCallRecord, SearchEvent
from .judge.semantic_judge import SemanticJudge, JudgeResult
from .judge.probe_evaluator import ProbeEvaluator, ProbeResult
from .rewards.two_stage_reward import TwoStageRewardFn, extract_final_answer
from .trainer.two_stage_trainer import TwoStageTrainer, TwoStageConfig, QADataset
from .trainer.tree_search_sampler import SearchNode, SearchGroup, TreeEpisode, TreeSearchSampler
from .trainer.tree_trainer import TreeTrainer, TreeTrainerConfig, compute_tree_loss
from .tools.tool_registry import ToolRegistry, ExecutionResult

__version__ = "0.3.0"
__all__ = [
    "QwenStyleAgent",
    "Episode",
    "ToolCallRecord",
    "SearchEvent",
    "SemanticJudge",
    "JudgeResult",
    "ProbeEvaluator",
    "ProbeResult",
    "TwoStageRewardFn",
    "extract_final_answer",
    "TwoStageTrainer",
    "TwoStageConfig",
    "QADataset",
    "SearchNode",
    "SearchGroup",
    "TreeEpisode",
    "TreeSearchSampler",
    "TreeTrainer",
    "TreeTrainerConfig",
    "compute_tree_loss",
    "ToolRegistry",
    "ExecutionResult",
]