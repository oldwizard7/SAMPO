# Math Agent API Evaluation

独立的评估系统，用于测试 API 模型（GPT-4o、Claude、DeepSeek 等）在数学问题上的表现。

## 功能特性

- ✅ **多轮交互**: 支持最多 5 轮对话，包含 code interpreter 工具使用
- ✅ **多模型支持**: OpenAI (GPT-4o, GPT-4o-mini), DeepSeek, Anthropic Claude
- ✅ **Sandbox 执行**: 安全的 Python 代码执行环境
- ✅ **完整轨迹**: 记录每一步的 input/output、code execution results
- ✅ **标准指标**: Accuracy, Pass@k, 平均 turns
- ✅ **批量评估**: 支持大规模数据集评估

## 快速开始

### 1. 设置 API Keys

```bash
export OPENAI_API_KEY="your-openai-api-key"
export DEEPSEEK_API_KEY="your-deepseek-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"
```

### 2. 运行评估

**使用 Shell 脚本（推荐）**：

```bash
# 基本用法
bash recipe/math_agent/eval/eval_math_agent.sh

# 指定模型和数据文件
bash recipe/math_agent/eval/eval_math_agent.sh gpt-4o ~/data/math/gsm8k_test.parquet

# 完整参数
bash recipe/math_agent/eval/eval_math_agent.sh \
    gpt-4o-mini \
    ~/data/math/test.parquet \
    8 \
    4
# 参数说明: model_name data_file batch_size num_samples
```

**直接使用 Python**：

```bash
python3 -m recipe.math_agent.eval.eval_math_agent_api \
    model_config.model_name=gpt-4o \
    data.val_files=~/data/math/test.parquet \
    data.val_batch_size=8 \
    actor_rollout_ref.rollout.val_kwargs.n=4
```

### 3. 查看结果

结果保存在 `outputs_math_agent/eval_results/` 目录：

- `result_<model>_seed<seed>_<timestamp>.pkl` - 完整轨迹（pickle 格式）
- `result_<model>_seed<seed>_<timestamp>.json` - 指标摘要（JSON 格式）
- `high_score_<model>_seed<seed>_<timestamp>.json` - 正确解答的轨迹

## 配置说明

### 主要配置参数

编辑 `config/base_eval.yaml` 和 `config/model_api.yaml` 来自定义评估：

**数据配置** (`base_eval.yaml`):
```yaml
data:
  val_files: ${oc.env:HOME}/data/math/test.parquet
  val_batch_size: 8
  seed: 42
```

**Agent 配置**:
```yaml
agent:
  max_turns: 5                    # 最大对话轮数
  sandbox_run_timeout: 3.0        # Code execution 超时（秒）
  append_final_answer_func: True  # 注入 final_answer() 函数
```

**模型配置** (`model_api.yaml`):
```yaml
model_config:
  model_name: gpt-4o              # 模型名称
  max_concurrency: 20             # API 并发数
  eval_system_prompt: |           # System prompt
    You are an expert mathematical problem solver...
```

**Reward Manager**:
```yaml
reward_model:
  reward_manager: math  # 选项: math, math_exec, code
```

## 支持的模型

在 `config/model_api.yaml` 中配置的模型：

| Provider | Model Name | Config Key |
|----------|-----------|------------|
| OpenAI   | GPT-4o    | `gpt-4o` |
| OpenAI   | GPT-4o-mini | `gpt-4o-mini` |
| DeepSeek | DeepSeek Chat | `deepseek-chat` |
| Anthropic | Claude 3.5 Sonnet | `claude-3-5-sonnet` |

添加新模型：在 `model_api.yaml` 中添加配置项。

## 输出指标

评估完成后输出以下指标：

- **Accuracy**: 整体准确率（0-1）
- **Pass@1**: 单次采样通过率
- **Pass@k**: k 次采样中至少 1 次正确的概率
- **Avg Turns**: 平均对话轮数

## 目录结构

```
eval/
├── __init__.py
├── eval_math_agent_api.py         # 主评估脚本
├── eval_math_agent.sh             # Shell 入口
├── README.md                      # 本文档
├── llm_agent/                     # API wrapper 模块
│   ├── __init__.py
│   ├── base_llm.py                # ConcurrentLLM (OpenAI, DeepSeek 支持)
│   ├── agent_proxy.py             # ApiCallingWrapperWg
│   └── evaluation_loop.py         # MathEvaluationLoop (核心逻辑)
└── config/                        # 配置文件
    ├── base_eval.yaml
    └── model_api.yaml
```

## 依赖复用

此评估模块复用 math_agent 训练模块的以下组件：

- `../agent_utils.py` - Code extraction patterns
- `../workers/reward_manager/` - Reward managers
- `../utils/rl_dataset/` - Dataset loading
- `sandbox.local_sandbox` - Code execution

## 故障排查

### API Key 未设置
```
ValueError: OpenAI API key not provided or missing in environment variables
```
**解决**: 设置对应的环境变量 `export OPENAI_API_KEY="..."`

### Sandbox 超时
```
Code execution result: interpreter timeout
```
**解决**: 增加 `agent.sandbox_run_timeout` 值（默认 3.0 秒）

### Import 错误
确保在 ARLArena 根目录下运行：
```bash
cd /home/ctong29/ARLArena
python3 -m recipe.math_agent.eval.eval_math_agent_api ...
```

## 高级用法

### 评估特定数据集

```bash
# GSM8K
python3 -m recipe.math_agent.eval.eval_math_agent_api \
    data.val_files=~/data/gsm8k/test.parquet \
    reward_model.reward_manager=math

# MATH dataset
python3 -m recipe.math_agent.eval.eval_math_agent_api \
    data.val_files=~/data/math/test.parquet \
    reward_model.reward_manager=math_exec
```

### 调整采样参数

```bash
# 高温度、更多样性
python3 -m recipe.math_agent.eval.eval_math_agent_api \
    model_info.gpt-4o.generation_kwargs.temperature=0.8 \
    actor_rollout_ref.rollout.val_kwargs.n=8
```

### 并发控制

```bash
# 降低并发（避免 rate limit）
python3 -m recipe.math_agent.eval.eval_math_agent_api \
    model_config.max_concurrency=10
```

## 贡献者

迁移自 shop_agent 的 evaluation 框架，适配 math_agent 需求。

## License

遵循 ARLArena 项目 License。
