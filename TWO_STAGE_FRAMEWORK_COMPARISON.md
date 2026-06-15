# 两阶段强化学习框架对比分析

## 概述

本文档对比分析新提出的两阶段强化学习框架与现有GRPO实现的差异，并制定迁移计划。

---

## 一、现有实现分析

### 1.1 架构概览

```
现有实现（单阶段GRPO）
├── Agent: Hermes风格function-calling
├── 奖励: 5通道奖励
│   ├── format_reward (±1)
│   ├── func_name_reward (-1到+1)
│   ├── args_reward (0到+1)
│   ├── outcome_reward (0或+1)
│   └── budget_reward (负值)
├── 训练: 单阶段GRPO
└── 评估: SemanticJudge混合投票
```

### 1.2 关键组件

#### Agent (qwen_style_agent.py)
- **格式**: Hermes风格function-calling
  ```
  <tool_call>{"name": "search", "arguments": {"query": "..."}}