# Think-Retriever

<div align="center">

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code Style: Black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**An intelligent retrieval-augmented generation agent with two-stage RL training**

</div>

## Overview

This repository implements **Think-Retriever**, an intelligent RAG system trained with **GRPO (Group Relative Policy Optimization)**. The agent learns to autonomously decide *when* and *how many times* to retrieve external knowledge during generation, optimizing for both answer correctness and retrieval efficiency.

### Key Features

- 🤖 **Token-level MDP formulation**: The full "Think → Search → Answer" loop is modeled as a Markov Decision Process
- 🎯 **Two-stage, three-channel reward**: Format + Tool-call + Answer rewards with dynamic retrieval budget constraints
- 🧠 **Semantic equivalence judge**: Hybrid voting replaces brittle exact-match for answer verification
- ⚡ **GRPO training**: Group-relative advantage normalization for stable on-policy RL
- 🔌 **Pluggable retrieval backends**: BM25, dense retrieval (FAISS), or custom APIs

---

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    GRPO Training Loop                    │
│                                                         │
│  Question ──► PolicyLM ──► [Think] [Search] [Answer]   │
│                  │               │                      │
│                  │         Retriever (BM25/Dense)       │
│                  │               │                      │
│                  └──── Reward ◄──┘                      │
│                           │                             │
│              ┌────────────┼────────────┐                │
│              ▼            ▼            ▼                │
│         FormatRwd   ToolCallRwd   AnswerRwd             │
│         (Stage 1)   (Stage 1)    (Stage 2)              │
└─────────────────────────────────────────────────────────┘
```

### MDP Formulation

| Component | Definition |
|-----------|-----------|
| **State** `s_t` | Generated token sequence + all retrieved documents so far |
| **Action** `a_t` | Next token from vocabulary `V` |
| **Transition** | Deterministic token append; tool calls trigger retrieval |
| **Reward** | Sparse; computed at end-of-sequence (two-stage) |
| **Policy** | Autoregressive LM `π_θ(a_t | s_t)` |

---

## Reward Design

### Stage 1 — Format & Tool-Call Rewards

| Channel | Condition | Reward |
|---------|-----------|--------|
| **Format** | Valid XML structure; unique trailing `<answer>` | `+1` |
| **Format** | Malformed XML or multiple `<answer>` blocks | `-1` |
| **Tool-call** | Model invoked retrieval ≥ 1 time | `+1` |
| **Tool-call** | No retrieval used | `0` |

### Stage 2 — Answer + Budget Constraint

| Channel | Condition | Reward |
|---------|-----------|--------|
| **Answer** | Semantically correct | `+1` |
| **Answer** | Wrong | `0` |
| **Budget** | Per extra retrieval call | `-0.1` |
| **Budget** | Correct answer bonus (retrieval refund) | `+0.5` |

**Total reward** = `format_reward + tool_reward + answer_reward + budget_reward`

---

## Installation

```bash
git clone https://github.com/your-org/think-retriever.git
cd think-retriever
pip install -e ".[dev]"
```

### Requirements

- Python ≥ 3.10
- PyTorch ≥ 2.1
- Transformers ≥ 4.40
- TRL ≥ 0.9
- FAISS (optional, for dense retrieval)

---

## Quick Start

### 1. Prepare Data

```bash
# Expected JSONL format: {"question": "...", "answer": "...", "context": "..."}
python scripts/prepare_data.py --input raw_data.json --output data/train.jsonl
```

### 2. Index Documents (Dense Retrieval)

```bash
python scripts/build_index.py \
    --corpus data/corpus.jsonl \
    --output data/faiss_index \
    --encoder BAAI/bge-base-en-v1.5
```

### 3. Train

```bash
# Single GPU
python train.py --config configs/default.yaml

# Multi-GPU (DeepSpeed ZeRO-2)
bash scripts/run_train.sh
```

### 4. Evaluate

```bash
python evaluate.py \
    --model_path outputs/checkpoint-final \
    --data data/test.jsonl \
    --retriever bm25 \
    --output_dir results/
```

---

## Configuration

Key parameters in `configs/default.yaml`:

```yaml
model:
  name_or_path: "Qwen/Qwen2.5-7B-Instruct"
  
grpo:
  group_size: 8           # G: samples per question
  learning_rate: 5e-7
  kl_coeff: 0.04          # KL divergence penalty
  clip_range: 0.2         # PPO clip ε
  
reward:
  format_correct: 1.0
  format_wrong: -1.0
  tool_call_used: 1.0
  tool_call_unused: 0.0
  answer_correct: 1.0
  retrieval_cost: -0.1    # per extra retrieval
  correct_refund: 0.5     # refund when answer correct
  
retriever:
  type: "bm25"            # or "dense"
  top_k: 3
```

---

## Project Structure

```
think-retriever/
├── src/think_retriever/
│   ├── agent/
│   │   ├── qwen_style_agent.py   # Core agent: rollout + tool dispatch
│   │   └── __init__.py
│   ├── rewards/
│   │   ├── two_stage_reward.py   # Two-stage reward function
│   │   └── __init__.py
│   ├── judge/
│   │   ├── semantic_judge.py     # Hybrid-voting semantic judge
│   │   └── probe_evaluator.py    # Probe evaluation for PSCA-SGPO
│   ├── trainer/
│   │   ├── two_stage_trainer.py  # Two-stage training loop
│   │   ├── tree_trainer.py       # Tree-search GRPO trainer
│   │   └── tree_search_sampler.py # Tree-structured exploration
│   ├── tools/
│   │   ├── tool_registry.py      # Tool registration and execution
│   │   └── __init__.py
│   └── utils/
│       ├── logging_utils.py
│       └── data_utils.py
├── configs/
│   └── default.yaml
├── data/
│   ├── hotpot/
│   └── sample/
├── train_two_stage.py
├── setup.py
└── requirements.txt
```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@misc{think-retriever-2025,
  title  = {Think-Retriever: An intelligent retrieval-augmented generation agent},
  year   = {2025},
  url    = {https://github.com/your-org/think-retriever}
}
```

---

## License

MIT License. See [LICENSE](LICENSE) for details.
