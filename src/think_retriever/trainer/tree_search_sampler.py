"""
tree_search_sampler.py
======================
Tree-Structured Search Sampler for Tree-Search GRPO

核心思想：
  1. 根节点 = Question
  2. 每个节点 = Think + Search + Content + ProbeAnswer
  3. 每个父节点展开G个子节点（G次不同的think+search）
  4. 同父亲节点构成一组，组内归一化优势
  5. 所有token（think + tool_call + answer）参与损失计算
  6. ProbeScore >= tau 时停止扩展，直接回答

树结构示例：
  Root (depth=0): Question
    Node_1_1: Think + Search + Content + Probe(Q=0.70)
    Node_1_2: Think + Search + Content + Probe(Q=0.25)
    Node_1_3: Think + Search + Content + Probe(Q=0.10)
    Node_1_4: Think + Search + Content + Probe(Q=0.95)
    └─ Group A: [Node_1_1, Node_1_2, Node_1_3, Node_1_4]
       └─ 组内优势 = (r_i - mean(r)) / (std(r) + epsilon)
    
    对 Q < tau 的节点继续扩展：
    Node_2_1 (parent=Node_1_2): Think + Search + Content + Probe(Q=0.55)
    Node_2_2 (parent=Node_1_2): Think + Search + Content + Probe(Q=0.40)
    Node_2_3 (parent=Node_1_2): Think + Search + Content + Probe(Q=0.75)
    Node_2_4 (parent=Node_1_2): Think + Search + Content + Probe(Q=0.30)
    └─ Group B: [Node_2_1, Node_2_2, Node_2_3, Node_2_4]
       └─ 组内优势基于共享父节点上下文计算
"""

from __future__ import annotations

import json
import logging
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch
from transformers import PreTrainedModel, PreTrainedTokenizerBase

logger = logging.getLogger(__name__)


# ── Special Tokens ────────────────────────────────────────────────────────────
_IM_START = "<|im_start|>"
_IM_END = "<|im_end|>"
_TOOL_CALL_START = "<tool_call>"
_TOOL_CALL_END = "</tool_call>"
_TOOL_RESPONSE_START = "<tool_response>"
_TOOL_RESPONSE_END = "</tool_response>"
_THINK_START = "<think>"
_THINK_END = "</think>"


# ── Data Structures ───────────────────────────────────────────────────────────

@dataclass
class SearchNode:
    """树中的一个搜索节点（一个Think + Search + ProbeAnswer）。"""
    node_id: str
    parent_id: Optional[str]
    depth: int  # 搜索深度（根节点=0）
    question: str
    
    # 生成的内容
    think_text: str = ""           # <think>...</think> 内部的推理
    search_query: str = ""          # tool_call 中的 query
    search_raw: str = ""            # 完整的 assistant 回复（含 think + tool_call）
    content_text: str = ""          # 搜索返回的内容
    probe_answer: str = ""          # Probe 回答（含 <think> + 答案）
    
    # 评估结果
    probe_score: float = 0.0        # Probe Value Q ∈ [0, 1]
    search_reward: float = 0.0      # r = (Q - Q_parent) - λ
    
    # token 信息（用于损失计算）
    input_ids: Optional[torch.Tensor] = None  # 完整prompt + 生成内容
    generation_start_pos: int = 0   # 从哪开始是模型生成的token
    think_token_range: Optional[Tuple[int, int]] = None  # think token 的位置范围
    search_token_range: Optional[Tuple[int, int]] = None  # search token 的位置范围
    answer_token_range: Optional[Tuple[int, int]] = None  # probe answer token 的位置范围
    
    # 组信息
    group_id: Optional[str] = None  # 同父节点组的ID
    advantage: float = 0.0          # 组内归一化后的优势
    
    # 终止标记
    is_terminal: bool = False       # 是否已停止扩展（Q >= tau 或 max_depth）
    final_answer: str = ""          # 如果是终端节点，最终答案


@dataclass
class SearchGroup:
    """一组共享同一父节点的搜索节点。"""
    group_id: str
    parent_id: str
    parent_probe_score: float       # Q_parent，用于计算 search reward
    depth: int
    nodes: List[SearchNode] = field(default_factory=list)
    is_final_group: bool = False    # 是否为最终组（包含足够好的答案）
    
    def compute_advantages(self, epsilon: float = 1e-8) -> None:
        """计算组内归一化优势。"""
        rewards = [n.search_reward for n in self.nodes]
        
        if len(rewards) <= 1:
            # 理论上不应发生，因为每个父节点固定展开G个
            for n in self.nodes:
                n.advantage = 0.0
            return
        
        mu = sum(rewards) / len(rewards)
        var = sum((r - mu) ** 2 for r in rewards) / len(rewards)
        sigma = var ** 0.5
        
        for n, r in zip(self.nodes, rewards):
            if sigma > epsilon:
                n.advantage = (r - mu) / (sigma + epsilon)
            else:
                n.advantage = 0.0


@dataclass
class TreeEpisode:
    """一次完整的树状搜索轨迹。"""
    root_question: str
    reference_answer: str
    max_depth: int
    branching_factor: int  # G: 每个父节点展开的子节点数
    stop_threshold: float  # tau: ProbeScore >= tau 时停止
    
    groups: List[SearchGroup] = field(default_factory=list)
    all_nodes: List[SearchNode] = field(default_factory=list)
    terminal_nodes: List[SearchNode] = field(default_factory=list)
    
    def total_nodes(self) -> int:
        return len(self.all_nodes)
    
    def max_reached_depth(self) -> int:
        return max((n.depth for n in self.all_nodes), default=0)


# ── Sampler Core ─────────────────────────────────────────────────────────────

class TreeSearchSampler:
    """
    树状搜索轨迹采样器。
    
    工作流程：
    1. 从 Question 作为根节点开始
    2. 对每个需要扩展的节点，生成 G 个不同的 (Think + Search)
    3. 对每个搜索结果做 Probe 评估，得到 Q_i
    4. 同父亲节点构成一组，计算组内优势
    5. 对 Q_i < tau 且 depth < max_depth 的节点，继续扩展下一层
    """
    
    def __init__(
        self,
        model: PreTrainedModel,
        tokenizer: PreTrainedTokenizerBase,
        retriever: Any,
        probe_evaluator: Any,
        branching_factor: int = 4,        # G: 每个父节点展开的子节点数
        max_depth: int = 3,               # 最大搜索深度
        stop_threshold: float = 0.9,      # tau: 停止扩展的阈值
        search_cost: float = 0.05,        # lambda: 搜索成本
        max_search_tokens: int = 128,     # think + search 最多生成多少token
        temperature: float = 1.0,
        top_p: float = 0.95,
        system_prompt: Optional[str] = None,
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer
        self.retriever = retriever
        self.probe_evaluator = probe_evaluator
        self.G = branching_factor
        self.max_depth = max_depth
        self.tau = stop_threshold
        self.lambda_cost = search_cost
        self.max_search_tokens = max_search_tokens
        self.temperature = temperature
        self.top_p = top_p
        
        self.system_prompt = system_prompt or (
            f"You are a helpful AI assistant with access to search tools. "
            f"Always start with <think> to reason about what to search. "
            f"Use <tool_call>{{\"name\": \"search\", \"arguments\": {{\"query\": \"...\"}}}}</tool_call> to search. "
            f"Be precise and focused in your search queries."
        )
    
    # ── Public API ───────────────────────────────────────────────────────────
    
    def sample_tree(
        self,
        question: str,
        reference_answer: str,
    ) -> TreeEpisode:
        """
        对一个问题生成完整的树状搜索轨迹。
        
        提前终止机制：
        - 一旦检测到 Q_i >= tau，跑完当前组的所有节点就停止
        - 保证每组都有 G 个节点用于计算优势
        
        Returns:
            TreeEpisode 包含所有节点、组和优势信息
        """
        episode = TreeEpisode(
            root_question=question,
            reference_answer=reference_answer,
            max_depth=self.max_depth,
            branching_factor=self.G,
            stop_threshold=self.tau,
        )
        
        # BFS 扩展树
        frontier: List[Tuple[int, Optional[str], float, str]] = []
        # (depth, parent_id, parent_probe_score, parent_context_string)
        
        # 根节点：只有问题，没有搜索
        initial_context = self._build_prompt(question, history=[])
        frontier.append((1, None, 0.0, initial_context))
        
        # 提前终止标志：一旦发现足够好的答案，标记需要停止
        early_stop = False
        
        while frontier and not early_stop:
            depth, parent_id, parent_q, parent_context = frontier.pop(0)
            
            if depth > self.max_depth:
                continue
            
            # 对当前父节点生成 G 个不同的 (think + search)
            group = self._expand_group(
                parent_id=parent_id or f"root-{uuid.uuid4().hex[:8]}",
                parent_probe_score=parent_q,
                parent_context=parent_context,
                question=question,
                reference_answer=reference_answer,
                depth=depth,
            )
            
            # 记录组和节点
            episode.groups.append(group)
            episode.all_nodes.extend(group.nodes)
            
            # 检查组内是否有足够好的答案
            group_has_good_answer = False
            best_node_in_group = None
            
            for node in group.nodes:
                if node.probe_score >= self.tau:
                    group_has_good_answer = True
                    if best_node_in_group is None or node.probe_score > best_node_in_group.probe_score:
                        best_node_in_group = node
            
            # 如果找到足够好的答案，标记提前终止
            if group_has_good_answer:
                early_stop = True
                # 标记该组为最终组
                group.is_final_group = True
                logger.info(
                    "Early stop triggered at depth %d: best probe_score=%.3f >= tau=%.3f",
                    depth, best_node_in_group.probe_score, self.tau
                )
            
            # 对每个节点，检查是否继续扩展
            for node in group.nodes:
                if node.probe_score >= self.tau:
                    # 停止扩展，记录最终答案
                    node.is_terminal = True
                    node.final_answer = node.probe_answer
                    episode.terminal_nodes.append(node)
                    logger.debug(
                        "Node %s STOP: probe_score=%.3f >= tau=%.3f",
                        node.node_id, node.probe_score, self.tau
                    )
                elif depth < self.max_depth and not early_stop:
                    # 继续扩展：把这个节点作为新的父节点
                    child_context = self._build_child_context(parent_context, node)
                    frontier.append((depth + 1, node.node_id, node.probe_score, child_context))
                else:
                    # 已达最大深度或提前终止，作为终端节点
                    node.is_terminal = True
                    node.final_answer = node.probe_answer
                    episode.terminal_nodes.append(node)
        
        logger.info(
            "Tree sampling done: %d nodes, %d groups, max depth=%d, early_stop=%s",
            episode.total_nodes(), len(episode.groups), episode.max_reached_depth(), early_stop
        )
        return episode
    
    # ── Internal: Group Expansion ────────────────────────────────────────────
    
    def _expand_group(
        self,
        parent_id: str,
        parent_probe_score: float,
        parent_context: str,
        question: str,
        reference_answer: str,
        depth: int,
    ) -> SearchGroup:
        """
        对一个父节点生成 G 个子节点，构成一个搜索组。
        
        Returns:
            SearchGroup: 包含 G 个搜索节点及其组内优势
        """
        group_id = f"group-{depth}-{parent_id}"
        group = SearchGroup(
            group_id=group_id,
            parent_id=parent_id,
            parent_probe_score=parent_probe_score,
            depth=depth,
        )
        
        for i in range(self.G):
            node = self._sample_one_search(
                parent_id=parent_id,
                parent_context=parent_context,
                question=question,
                reference_answer=reference_answer,
                depth=depth,
                child_index=i,
                parent_probe_score=parent_probe_score,
            )
            node.group_id = group_id
            group.nodes.append(node)
        
        # 计算组内优势
        group.compute_advantages()
        
        logger.debug(
            "Group %s (depth=%d): rewards=%s, advantages=%s",
            group_id, depth,
            [f"{n.search_reward:.3f}" for n in group.nodes],
            [f"{n.advantage:.2f}" for n in group.nodes],
        )
        
        return group
    
    def _sample_one_search(
        self,
        parent_id: str,
        parent_context: str,
        question: str,
        reference_answer: str,
        depth: int,
        child_index: int,
        parent_probe_score: float,
    ) -> SearchNode:
        """
        采样一个子节点：Think + Search + Content + ProbeAnswer
        
        流程：
        1. 给模型喂入 parent_context（含历史搜索信息）
        2. 让模型生成 Think + Tool_call（search query）
        3. 执行搜索，得到 Content
        4. 强制模型基于（parent_context + Content）生成 Answer
        5. 评估 Answer 质量，得到 ProbeScore
        6. 计算 Search Reward = (Q - Q_parent) - λ
        """
        node = SearchNode(
            node_id=f"node-{depth}-{parent_id}-{child_index}",
            parent_id=parent_id,
            depth=depth,
            question=question,
        )
        
        # Step 1 & 2: 生成 Think + Search
        search_text, think_text, search_query, (input_ids, gen_start) = \
            self._generate_thinking_and_search(parent_context, depth)
        
        node.search_raw = search_text
        node.think_text = think_text
        node.search_query = search_query
        node.input_ids = input_ids
        node.generation_start_pos = gen_start
        
        # 记录 think 和 search 的 token 范围
        tokenized_full = self.tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=False)
        node.think_token_range = self._find_token_range(input_ids[0], _THINK_START, _THINK_END)
        node.search_token_range = self._find_token_range(input_ids[0], _TOOL_CALL_START, _TOOL_CALL_END)
        
        # Step 3: 执行搜索
        content_text = self._do_search(search_query)
        node.content_text = content_text
        
        # Step 4: Probe - 强制模型基于当前信息回答（带思考过程）
        probe_prompt = self._build_probe_prompt(parent_context, content_text)
        probe_raw = self._generate_probe_answer(probe_prompt, question, with_thinking=True)
        node.probe_answer = probe_raw  # 完整的 <think> + 答案

        # 提取纯答案用于评估
        probe_answer_only = self._extract_answer_only(probe_raw)

        # 记录 probe answer 的 token 范围
        probe_input_ids = self.tokenizer(
            probe_prompt + probe_raw, return_tensors="pt"
        ).input_ids.to(self.model.device)
        node.answer_token_range = (
            self.tokenizer(probe_prompt, return_tensors="pt").input_ids.shape[1],
            probe_input_ids.shape[1],
        )

        # Step 5: Probe 评估（使用纯答案，不包含 <think>）
        node.probe_score = self.probe_evaluator.evaluate(probe_answer_only, reference_answer)
        
        # Step 6: 计算 search reward
        node.search_reward = (node.probe_score - parent_probe_score) - self.lambda_cost
        
        logger.debug(
            "  Node %s: q=%.2f -> q_new=%.2f, r=%.3f, query='%s'",
            node.node_id, parent_probe_score, node.probe_score,
            node.search_reward, search_query[:50]
        )
        
        return node
    
    # ── Internal: Generation ───────────────────────────────────────────────
    
    def _generate_thinking_and_search(
        self,
        parent_context: str,
        depth: int,
    ) -> Tuple[str, str, str, Tuple[torch.Tensor, int]]:
        """
        生成 Think + Tool_call。
        
        Returns:
            (完整生成文本, think内容, search query, (input_ids, gen_start_pos))
        """
        # 准备输入
        prompt = parent_context + f"{_IM_START}assistant\n"
        
        encoded = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encoded.input_ids.to(self.model.device)
        gen_start_pos = input_ids.shape[1]
        
        # 生成（最多 max_search_tokens 个 token，遇到 tool_call_end 停止）
        with torch.inference_mode():
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=self.max_search_tokens,
                do_sample=True,
                temperature=self.temperature,
                top_p=self.top_p,
                pad_token_id=self.tokenizer.eos_token_id,
                stopping_criteria=None,  # 后处理找停止标志
            )
        
        full_ids = output_ids[0]
        generated_ids = full_ids[gen_start_pos:]
        generated_text = self.tokenizer.decode(generated_ids, skip_special_tokens=False)
        
        # 提取 think 内容
        think_match = generated_text.find(_THINK_START)
        think_end = generated_text.find(_THINK_END)
        think_content = ""
        if think_match >= 0 and think_end > think_match:
            think_content = generated_text[think_match + len(_THINK_START): think_end].strip()
        
        # 提取 search query
        search_query = self._extract_search_query(generated_text)
        if not search_query:
            # fallback: 从生成文本中提取有意义的查询
            search_query = self._fallback_query(generated_text, depth)
        
        return generated_text, think_content, search_query, (full_ids.unsqueeze(0), gen_start_pos)
    
    def _do_search(self, query: str) -> str:
        """执行实际搜索。"""
        try:
            result = self.retriever.search(query)
            if isinstance(result, str):
                return result
            if hasattr(result, 'content'):
                return result.content
            if isinstance(result, dict) and 'content' in result:
                return str(result['content'])
            return str(result)
        except Exception as e:
            logger.warning("Search failed for query='%s': %s", query[:50], e)
            return "[Search failed: no results]"
    
    def _build_probe_prompt(self, parent_context: str, content_text: str) -> str:
        """
        构建 Probe 评估的 prompt（强制回答）。

        使用与 Stage 1 一致的格式，包括 <tool_response> 和 <answer> 标签。
        通过明确指令强制模型基于已有信息直接回答，并输出标准格式。
        """
        # 与 Stage 1 保持一致：使用 <tool_response> 包裹搜索结果
        tool_response = (
            f"{_TOOL_RESPONSE_START}"
            f"{{\"success\": true, \"result\": {json.dumps(content_text)}}}"
            f"{_TOOL_RESPONSE_END}"
        )
        return (
            f"{parent_context}"
            f"{_IM_START}user\n"
            f"{tool_response}\n"
            f"Based on the previous contents, answer the question.\n"
            f"Wrap your answer in <answer>...</answer> tags.\n"
            f"{_IM_END}\n"
            f"{_IM_START}assistant\n"
        )
    
    def _generate_probe_answer(
        self,
        prompt: str,
        question: str,
        with_thinking: bool = False,
    ) -> str:
        """
        基于 prompt 生成 Probe 回答。

        Parameters
        ----------
        prompt: 输入提示词
        question: 原始问题（用于调试日志）
        with_thinking: 是否保留 <think> 标签（默认 False，保持向后兼容）

        Returns
        -------
        如果 with_thinking=True: 返回 <think>...</think> + 答案
        如果 with_thinking=False: 只返回答案部分
        """
        with torch.inference_mode():
            input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.model.device)
            output_ids = self.model.generate(
                input_ids,
                max_new_tokens=100,
                do_sample=False,  # 评估时用确定性生成
                temperature=0.0,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            answer_ids = output_ids[0, input_ids.shape[1]:]
            raw_output = self.tokenizer.decode(answer_ids, skip_special_tokens=False)

            if with_thinking:
                # 保留 <think>...</think> 标签和答案
                return raw_output.strip()
            else:
                # 去掉 <think> 标签，只保留答案
                return self._extract_answer_only(raw_output)

    def _extract_answer_only(self, text: str) -> str:
        """
        从包含 <think>...</think> 和 <answer>...</answer> 的文本中提取答案部分。

        策略：
        1. 先去掉 <think>...</think> 标签及其内容
        2. 提取 <answer>...</answer> 标签内的内容（如果存在）
        3. 如果没有 <answer> 标签，返回剩余文本
        """
        # 去掉 <think>...</think> 标签及其内容
        cleaned = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        
        # 尝试提取 <answer>...</answer> 标签内的内容
        answer_match = re.search(r'<answer>(.*?)</answer>', cleaned, re.DOTALL)
        if answer_match:
            answer = answer_match.group(1).strip()
            return answer if answer else cleaned.strip()
        
        # 如果没有 <answer> 标签，返回清理后的文本
        return cleaned.strip()

    def _generate_answer(self, prompt: str, question: str) -> str:
        """基于 prompt 生成答案（用于 Probe 评估和最终回答）。"""
        return self._generate_probe_answer(prompt, question, with_thinking=False)
    
    # ── Internal: Prompt Building ───────────────────────────────────────────
    
    def _build_prompt(self, question: str, history: List[str]) -> str:
        """构建完整对话 prompt（含系统提示和历史信息）。"""
        prompt = f"{_IM_START}system\n{self.system_prompt}{_IM_END}\n"
        prompt += f"{_IM_START}user\n{question}{_IM_END}\n"
        for h in history:
            prompt += h
        return prompt
    
    def _build_child_context(self, parent_context: str, node: SearchNode) -> str:
        """
        把一个节点的搜索结果追加到对话历史中，作为其子节点的 prompt。
        
        生成的上下文格式：
            [原parent_context]
            <|im_start|>assistant
            <think>...思考内容...</think>
            <tool_call>{"name": "search", ...}</tool_call><|im_end|>
            <|im_start|>user
            <tool_response>{"result": "..."}</tool_response><|im_end|>
        """
        child_context = parent_context
        child_context += f"{node.search_raw}{_IM_END}\n"
        child_context += f"{_IM_START}user\n"
        child_context += f"{_TOOL_RESPONSE_START}{{\"result\": {json.dumps(node.content_text[:500])}}}{_TOOL_RESPONSE_END}{_IM_END}\n"
        return child_context
    
    # ── Internal: Parsing Helpers ──────────────────────────────────────────
    
    def _extract_search_query(self, generated_text: str) -> str:
        """从生成文本中提取 search query。"""
        import re
        
        # 找 <tool_call>...</tool_call>
        tool_match = re.search(
            r'<tool_call>\s*(.*?)\s*</tool_call>',
            generated_text, re.DOTALL
        )
        if not tool_match:
            return ""
        
        try:
            tool_payload = json.loads(tool_match.group(1))
        except json.JSONDecodeError:
            return ""
        
        if isinstance(tool_payload, dict):
            args = tool_payload.get("arguments", {})
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except json.JSONDecodeError:
                    pass
            if isinstance(args, dict):
                query = args.get("query", "")
                if isinstance(query, str) and query.strip():
                    return query.strip()
        return ""
    
    def _fallback_query(self, generated_text: str, depth: int) -> str:
        """当解析失败时的回退查询（从生成文本中提取最后几个有意义的词）。"""
        cleaned = generated_text.replace("<", " ").replace(">", " ")
        words = [w for w in cleaned.split() if len(w) > 2]
        if words:
            return " ".join(words[-5:])
        return f"information check depth {depth}"
    
    def _find_token_range(
        self,
        input_ids: torch.Tensor,
        start_marker: str,
        end_marker: str,
    ) -> Optional[Tuple[int, int]]:
        """
        找到 start_marker 到 end_marker 在 token 序列中的范围。
        
        Returns: (start_pos, end_pos) 或 None（如果没找到）
        """
        text = self.tokenizer.decode(input_ids.tolist(), skip_special_tokens=False)
        
        start_idx = text.find(start_marker)
        end_idx = text.find(end_marker)
        
        if start_idx < 0 or end_idx < 0 or end_idx <= start_idx:
            return None
        
        # 将字符位置映射回 token 位置（通过二分编码）
        prefix_start = text[:start_idx]
        prefix_end = text[:end_idx + len(end_marker)]
        
        tokens_start = len(self.tokenizer.encode(prefix_start))
        tokens_end = len(self.tokenizer.encode(prefix_end))
        
        return (tokens_start, tokens_end)
