# 主流框架工具调用格式研究报告

## 1. OpenAI Function Calling 格式

### 官方标准格式

```json
// 工具定义
{
  "type": "function",
  "function": {
    "name": "get_weather",
    "description": "获取指定城市的天气",
    "parameters": {
      "type": "object",
      "properties": {
        "location": {
          "type": "string",
          "description": "城市名称"
        }
      },
      "required": ["location"]
    }
  }
}

// 模型输出 (tool_calls)
{
  "tool_calls": [
    {
      "id": "call_abc123",
      "type": "function",
      "function": {
        "name": "get_weather",
        "arguments": "{\"location\": \"北京\"}"
      }
    }
  ]
}

// 工具结果 (tool role)
{
  "role": "tool",
  "tool_call_id": "call_abc123",
  "content": "北京今天晴天，25度"
}
```

### OpenAI 的 Prefix Caching 支持

OpenAI API 本身不暴露 KV Cache，但其格式设计考虑了效率：

1. **Tool Schema 预定义**: 工具定义在请求开始时发送
2. **参数 JSON Schema**: 标准化参数格式
3. **Tool Call ID**: 唯一标识，用于匹配结果

## 2. Anthropic Claude Tool Use 格式

### 官方标准格式

```json
// 工具定义
{
  "name": "weather",
  "description": "获取天气信息",
  "input_schema": {
    "type": "object",
    "properties": {
      "location": {"type": "string"}
    },
    "required": ["location"]
  }
}

// 模型输出 (tool_use block)
{
  "content": [
    {
      "type": "tool_use",
      "id": "toolu_abc123",
      "name": "weather",
      "input": {"location": "北京"}
    }
  ]
}

// 工具结果 (tool_result block)
{
  "content": [
    {
      "type": "tool_result",
      "tool_use_id": "toolu_abc123",
      "content": "北京今天晴天，25度"
    }
  ]
}
```

### Claude 的特点

1. **Block 结构**: 使用明确的块结构标记工具调用和结果
2. **独立 Input Schema**: 与 OpenAI 类似但结构略有不同
3. **Content 作为数组**: 支持多模态内容

## 3. Google Gemini Function Calling 格式

```json
// 工具定义
{
  "name": "get_weather",
  "description": "获取天气",
  "parameters": {
    "type": "object",
    "properties": {
      "location": {"type": "string"}
    }
  }
}

// 函数调用
{
  "functionCall": {
    "name": "get_weather",
    "args": {"location": "北京"}
  }
}

// 函数响应
{
  "functionResponse": {
    "name": "get_weather",
    "response": {"weather": "晴天"}
  }
}
```

## 4. Together AI / VLLM Extended JSON Schema

针对 KV Cache 优化的格式：

```json
// Tool Call with Parallel
{
  "tool_calls": [
    {
      "id": "0",
      "name": "search",
      "arguments": {"query": "capital of France"}
    }
  ]
}

// Tool Result
{
  "tool_results": [
    {
      "id": "0",
      "result": "Paris"
    }
  ]
}
```

## 5. 主流框架共同点分析

### 格式对比表

| 特性 | OpenAI | Anthropic | Gemini | 本项目 XML |
|------|--------|-----------|--------|-----------|
| 工具定义 | JSON Schema | Input Schema | Parameters | N/A |
| 调用标识 | id | id | - | - |
| 参数格式 | JSON string | Object | Object | XML 标签 |
| 结果格式 | tool role | tool_result | functionResponse | XML 标签 |
| 批量调用 | tool_calls[] | tool_use[] | functionCall | 多次调用 |
| 多模态 | 支持 | 支持 | 支持 | 纯文本 |

### 共同模式

1. **工具定义与执行分离**: 工具 schema 预定义，执行时传入参数
2. **唯一标识符**: 每个工具调用有唯一 ID 用于匹配
3. **结构化参数**: 使用 JSON/Object 格式传递参数
4. **独立结果通道**: 工具结果通过专门的消息类型返回

## 6. 推荐的标准 JSON 格式

基于主流框架的最佳实践，建议采用以下格式：

### 工具调用 (Tool Call)

```json
{
  "tool_calls": [
    {
      "id": "call_001",
      "name": "search",
      "arguments": {
        "query": "搜索内容"
      }
    }
  ]
}
```

### 工具结果 (Tool Result)

```json
{
  "tool_results": [
    {
      "id": "call_001",
      "result": "搜索结果内容"
    }
  ]
}
```

### 最终答案 (Answer)

```json
{
  "answer": "这是最终答案"
}
```

### 完整对话示例

```
User: 什么是人工智能？

Assistant: [
  {
    "tool_calls": [
      {
        "id": "call_001",
        "name": "search",
        "arguments": {"query": "人工智能定义"}
      }
    ]
  }
]

Tool: [
  {
    "tool_results": [
      {
        "id": "call_001",
        "result": "人工智能是计算机科学的一个分支..."
      }
    ]
  }
]

Assistant: [
  {
    "answer": "人工智能是计算机科学的一个分支，致力于创造智能机器。"
  }
]
```

## 7. 与 Prefix Caching 的兼容性

### 主流框架的做法

1. **OpenAI**: 服务器端处理，API 不暴露 KV Cache
2. **Anthropic**: 使用特殊的 content block 格式
3. **Together AI**: 支持 extended JSON schema，带 parallel tool calls

### 推荐做法

对于需要优化 Prefix Caching 的场景：

1. **工具定义放 System Prompt**: 只计算一次
2. **使用标准 JSON 格式**: 保持兼容性
3. **批量处理**: 一次返回多个工具调用结果

```json
// System Prompt (缓存)
{
  "tools": [
    {"name": "search", "description": "搜索"},
    {"name": "calculator", "description": "计算"}
  ],
  "format": "tool_calls with arguments"
}

// 每次调用
{"tool_calls": [{"id": "1", "name": "search", "arguments": {"query": "..."}}]}
{"tool_results": [{"id": "1", "result": "..."}]}
{"answer": "..."}
```

## 8. 结论

### 主流趋势

1. **JSON 格式是标准**: 所有主流框架都使用 JSON
2. **结构化参数**: 使用 JSON Schema 定义参数
3. **唯一 ID 追踪**: 每个调用有唯一标识符
4. **批量支持**: 一次可以触发多个工具调用

### 建议

1. **保持 JSON 格式**: 改回标准 JSON，不要用 XML
2. **使用主流格式**: 遵循 OpenAI/Anthropic 的设计
3. **添加工具定义**: 在 System Prompt 中预定义工具
4. **支持批量调用**: 一次返回多个工具结果

### 最终推荐格式

```json
// 工具调用
{"tool_calls": [{"id": "call_001", "name": "search", "arguments": {"query": "..."}}]}

// 工具结果
{"tool_results": [{"id": "call_001", "result": "..."}]}

// 最终答案
{"answer": "最终答案"}
```
