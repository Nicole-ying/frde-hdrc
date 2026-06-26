# Eureka 式流程修正说明

## 1. Baseline 流程

以 LunarLander-v3 为例，baseline 现在定义为：使用环境官方 `step()` 中的原始奖励函数，PPO 使用 RL Zoo 风格超参数训练一轮并评估原始环境回报。代码中对应 `reward_program=None`，即不替换环境 reward；PPO 参数来自 `ExperimentConfig.ppo_kwargs()`，默认包含 `learning_rate=2.5e-4`、`n_steps=1024`、`n_epochs=4`、`gamma=0.999`、`gae_lambda=0.98`、`ent_coef=0.01`。

## 2. FDRE 初始奖励生成流程

FDRE 不再直接把环境名和人工奖励知识交给 LLM，而是采用 Eureka 式输入：

- `code/contexts/lunarlander/task_description.txt`
- `code/contexts/lunarlander/step_masked.py`
- `code/src/llm_reward_evolver/prompts.py`

其中 `step_masked.py` 来自 Gymnasium LunarLander-v3 的 `step()` 源码，但官方 reward 计算块已被 `<OFFICIAL_REWARD_MASKED>` 替换，且官方 shaping 公式已删除。LLM 只能根据 task description、masked step source 和 prompt 生成 `compute_reward()`。

## 3. 防止奖励泄露

修正后默认 `allow_original_reward_in_reward=false`，因此生成奖励函数不能引用 `original_reward`。`RewardEvolver` 会在训练前检查候选代码，如果出现 `original_reward`、`OFFICIAL_REWARD_MASKED`、`masked_reward` 或 `prev_shaping` 等泄露信号，会直接判定为无效并要求重新生成。baseline 仍然可以使用官方原始奖励，因为 baseline 的定义就是“官方 reward + PPO 训练”；FDRE 训练奖励不能使用官方 reward。

## 4. 迭代与 Memory

每一轮训练后，系统会把本轮 reward 的得分、成功率、episode 长度、成功/失败 verdict 和经验 lesson 写入：

- `reward_search_memory.json`

下一轮 refine prompt 会读取这些 memory，告诉 LLM 哪些 reward 结构是失败经验，哪些候选应保守继承。这使流程从简单“重新问 LLM”变成带经验记忆的 reward 空间搜索。

## 5. Smoke 复验

本轮已使用 `configs/lunarlander_smoke_verify.json` 完成 smoke 复验。复验结果显示：

- Ollama `qwen2.5-coder:7b` 可正常生成 reward。
- 生成 reward 未使用 `original_reward`。
- 未出现 `OFFICIAL_REWARD_MASKED`、`masked_reward`、`prev_shaping` 泄露。
- PPO 训练和评估链路正常结束。
- `reward_error_count=0`，`interrupted=false`。
- `reward_search_memory.json` 正常记录失败经验。

相关记录在：

- `data/lunarlander_smoke_verify_reward_iter_0.py`
- `data/lunarlander_smoke_verify_feedback_iter_0.txt`
- `data/lunarlander_smoke_verify_reward_search_memory.json`
