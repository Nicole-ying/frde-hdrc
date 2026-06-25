# 代码交付说明

本目录为 FDRE-HRDC 实验代码快照，包含核心源码、实验脚本、配置文件、测试文件和依赖声明。

## 当前代码口径

主 FDRE-HRDC 路径已经按 agent 自主奖励搜索整理：`RewardEvolver` 只向大模型提供通用 observation/action space、原始奖励、训练反馈、运行错误和评估统计，不再注入 LunarLander 的人工状态语义或手写奖励规则。`suite.py` 中的主 `fdre` 方法也改为调用 `RewardEvolver` 迭代搜索；固定奖励程序仅作为 legacy/canonical 对照和消融参考保留。

## 本轮框架修复

- PPO 默认参数已改为 RL Zoo 风格配置，降低同一奖励函数在不同 PPO seed 下的额外训练方差。
- 新增 `src/llm_reward_evolver/diagnostics.py`，在训练前检查 `original_reward` anchor，拦截 `original_reward * 0.001`、变量形式 tiny anchor 和完全不用原始奖励的候选函数。
- Prompt 中加入 anchor 硬约束：`original_reward` 缩放不得低于 `0.5x`，推荐 `reward = original_reward + shaping_terms`。
- `RewardEvolver` 增加结构性重启机制，连续低分时不再被低质量 best_code 锁死。
- 主 FDRE 多 seed 实验增加共享 reward pool，高分 seed 的候选可以作为失败 seed 的重启起点。
- LunarLander 配置已改为黑盒任务描述，避免把平稳降落、直立、燃料、触地等领域语义写入 LLM 输入。

## 目录

- `src/llm_reward_evolver/`：奖励函数生成、自进化、反馈诊断、训练封装和报告生成模块。
- `scripts/run_experiment.py`：实验入口，支持 `mock`、`ollama`、`deepseek` 调用方式。
- `scripts/generate_publication_figures.py`：论文版单图生成脚本。
- `configs/lunarlander_paper.json`：LunarLander-v3 对比实验配置。
- `configs/lunarlander_robust_1m.json`：10 seed、1M timesteps 的稳健复验配置。
- `configs/lunarlander_smoke_verify.json`：交付前快速 smoke 复验配置。
- `contexts/lunarlander/task_description.txt`：Eureka 式任务描述输入。
- `contexts/lunarlander/step_masked.py`：官方 reward 已 mask 的环境 step 源码输入。
- `src/llm_reward_evolver/eureka_context.py`：读取 task/step 上下文并 mask reward 的模块。
- `src/llm_reward_evolver/memory.py`：记录成功/失败经验的 agent memory 模块。
- `tests/`：基础奖励函数测试。

## 常用命令

```powershell
python scripts/run_experiment.py --config configs/lunarlander_paper.json --llm-provider ollama --llm-model qwen2.5-coder:7b --suite
python scripts/run_experiment.py --config configs/lunarlander_robust_1m.json --llm-provider ollama --llm-model qwen2.5-coder:7b --suite
python scripts/run_experiment.py --config configs/lunarlander_smoke_verify.json --llm-provider ollama --llm-model qwen2.5-coder:7b
python scripts/generate_publication_figures.py
python -m pytest tests
```

当前交付数据对应 LunarLander-v3：1M 最终评估平均得分 `226.04 +/- 6.31`，三个 seed 均超过 200 分；reward runtime error 为 `0`，由奖励函数错误导致的训练中断为 `0`。
