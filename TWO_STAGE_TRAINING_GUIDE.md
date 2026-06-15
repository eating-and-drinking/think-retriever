# Two-Stage RL Training Guide

## Overview

This guide explains how to use the new two-stage training framework implementing:
- **Stage 1: RPA** (ReAct Protocol Alignment)
- **Stage 2: PSCA-SGPO** (Probe-based Search Credit Assignment)

## Quick Start

```bash
# Run two-stage training
python train_two_stage.py --config configs/default.yaml

# With overrides
python train_two_stage.py --config configs/default.yaml \
    --stage1_epochs 2 \
    --stage2_epochs 1 \
    --output_dir outputs/my_experiment/
```

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    Two-Stage Training                             │
│                                                                  │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Stage 1: RPA (ReAct Protocol Alignment)                   │   │
│  │                                                           │   │
│  │  Question → Think → Search → Content → Answer            │   │
│  │                                                           │   │
│  │  Rewards:                                                 │   │
│  │    • Format Reward (±1)                                   │   │
│  │    • Protocol Reward (0-1)                                │   │
│  │    • Budget Reward (-penalty)                            │   │
│  │    • Answer Reward (0 or +1)                             │   │
│  │                                                           │   │
│  │  Update: Entire trajectory (GRPO)                        │   │
│  └──────────────────────────────────────────────────────────┘   │
│                            ↓                                      │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │ Stage 2: PSCA-SGPO (Probe-based Search Credit Assignment)│   │
│  │                                                           │   │
│  │  Search → Probe(Q_i) → ΔQ - λ → State-aware Advantage    │   │
│  │                                                           │   │
│  │  State = (depth, bucket(Q))                              │   │
│  │  Advantage = (r_i - μ_{d,b}) / (σ_{d,b} + ε)            │   │
│  │                                                           │   │
│  │  Update: Only search tokens                              │   │
│  └──────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

## Qwen-Style Format

The new `QwenStyleAgent` uses Qwen's format:

```xml
<|im_start|>user
What is the capital of France?<|im_end|>
<|im_start|>assistant
<think>
The user is asking about the capital of France. I need to search for this.
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

## Configuration

### Two-Stage Training Settings

```yaml
# Stage 1: RPA
stage1:
  epochs: 2                      # Number of epochs
  group_size: 8                 # Samples per question
  lr: 5.0e-7                    # Learning rate

# Stage 2: PSCA-SGPO
stage2:
  epochs: 1                      # Number of epochs
  group_size: 4                 # Samples per question
  lr: 1.0e-7                    # Smaller LR for fine-tuning

# PSCA-SGPO Parameters
psca_sgpo:
  search_cost: 0.05             # λ: cost per search
  early_stop_threshold: 0.9     # τ: stop when Q >= τ
  epsilon: 1.0e-8               # Numerical stability
```

## Probe Mechanism

The Probe mechanism evaluates knowledge state after each search:

1. **After search**: Force model to answer based on current knowledge
2. **Evaluate**: Get Probe Value Q_i ∈ [0, 1]
3. **Compute reward**: r_i = (Q_i - Q_{i-1}) - λ

### Example

| Search | Probe Value Q_i | ΔQ | Search Reward r_i |
|--------|----------------|-----|-------------------|
| 0 (initial) | 0.0 | - | - |
| 1 | 0.30 | 0.30 | 0.30 - 0.05 = **0.25** |
| 2 | 0.65 | 0.35 | 0.35 - 0.05 = **0.30** |
| 3 | 0.90 | 0.25 | 0.25 - 0.05 = **0.20** |

## State-Aware Grouping

Search events are grouped by:
- **depth**: Search depth (1, 2, 3, ...)
- **q_bucket**: Knowledge state bucket (B0-B4)

### Q Buckets

| Bucket | Range | Description |
|--------|-------|-------------|
| B0 | [0.0, 0.2) | Very low knowledge |
| B1 | [0.2, 0.4) | Low knowledge |
| B2 | [0.4, 0.6) | Medium knowledge |
| B3 | [0.6, 0.8) | Good knowledge |
| B4 | [0.8, 1.0] | High knowledge |

### Advantage Calculation

For each state (depth, bucket), compute:

```
μ_{d,b} = mean(r_i for events in state (d,b))
σ_{d,b} = std(r_i for events in state (d,b))
A_i = (r_i - μ_{d,b}) / (σ_{d,b} + ε)
```

This compares search quality within similar states, not globally.

## New Components

### QwenStyleAgent
- Location: `src/think_retriever/agent/qwen_style_agent.py`
- Features:
  - Qwen-style format with `<|im_start|>`/`<|im_end|>`
  - Built-in Probe mechanism support
  - Search event tracking

### ProbeEvaluator
- Location: `src/agentic_rag/judge/probe_evaluator.py`
- Features:
  - Evaluates knowledge state after each search
  - Reuses SemanticJudge for answer quality

### StateAwareGrouper
- Location: `src/agentic_rag/trainer/state_aware_grouper.py`
- Features:
  - Groups search events by (depth, q_bucket)
  - Computes state-aware advantages

### TwoStageRewardFn
- Location: `src/agentic_rag/rewards/two_stage_reward.py`
- Features:
  - Stage 1: Format + Protocol + Budget + Answer rewards
  - Stage 2: Search rewards from Probe mechanism

### TwoStageTrainer
- Location: `src/agentic_rag/trainer/two_stage_trainer.py`
- Features:
  - Orchestrates both training stages
  - Stage 1: Standard GRPO on entire trajectory
  - Stage 2: Search token loss with state-aware advantages

## Training Flow

```python
from agentic_rag import (
    QwenStyleAgent,
    TwoStageTrainer,
    TwoStageRewardFn,
    ProbeEvaluator,
    StateAwareGrouper,
)

# Build components
agent = QwenStyleAgent(...)
reward_fn = TwoStageRewardFn(...)
probe_eval = ProbeEvaluator(...)
state_grouper = StateAwareGrouper(...)

# Create trainer
trainer = TwoStageTrainer(
    model=model,
    tokenizer=tokenizer,
    agent=agent,
    reward_fn=reward_fn,
    probe_evaluator=probe_eval,
    state_grouper=state_grouper,
    config=config,
)

# Train (runs both stages automatically)
trainer.train(train_dataset, eval_dataset)
```

## Comparison: Old vs New

| Aspect | Old (Single-Stage) | New (Two-Stage) |
|--------|---------------------|-----------------|
| Format | Hermes JSON | Qwen XML |
| Training | Single-stage GRPO | Two-stage (RPA + PSCA-SGPO) |
| Rewards | 5-channel (format+func+args+outcome+budget) | Stage 1: 4-channel<br>Stage 2: Probe-based |
| Update Scope | Entire trajectory | Stage 1: Entire<br>Stage 2: Search tokens only |
| Search Credit | Outcome-based | Probe-based + State-aware |

## Next Steps

1. **Prepare data**: Ensure your data is in JSONL format with `question` and `answer` fields
2. **Configure**: Update `configs/default.yaml` with your settings
3. **Train**: Run `python train_two_stage.py --config configs/default.yaml`
4. **Evaluate**: Use the existing evaluation scripts
