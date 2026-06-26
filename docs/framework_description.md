# FDRE-HRDC 框架流程说明（论文用）

## 一、创新点定位

### 创新一：HRDC — 层次化奖励动态组合（算法核心）

传统的 LLM 奖励生成方法（EUREKA、Text2Reward、CARD）允许 LLM 自由设计权重结构。这种自由度导致生成质量高度随机——同一 prompt 下，LLM 可能生成简洁的两阶段权重、九维连续插值或三角窗归一化，搜索效率严重依赖初始生成质量。

HRDC（Hierarchical Reward Decomposition with Dynamic Weights）强制要求每个奖励组件拥有独立的、随训练进度（training_progress）演化的阶段权重，形成"早期探索→中期平衡→晚期精度"的结构先验。具体而言：

- 每个奖励组件（如运动信号、稳定性惩罚、接触奖励）在不同训练阶段获得独立权重，而非共享单一的 early/late 系数
- 权重随 training_progress 从 [0,1] 演化，早期侧重探索性信号，晚期侧重精度信号
- 这一结构作为 Prompt 中的硬约束传递给 LLM，LLM 在生成代码时必须遵循

HRDC 的核心贡献在于：将"如何组织奖励组件"这一设计空间从 LLM 的自由创作转变为有结构的参数化搜索。这不仅降低了搜索方差，还使得 Agent 的骨架诊断能够对每个组件进行独立的因果分析。

### 创新二：Autonomous Reward Design Agent（系统架构）

我们将奖励函数搜索重新定义为 Agent 决策问题。与现有 LLM 奖励设计方法（它们本质上是"Prompt→代码→训练→反馈→Prompt"的算法管道）不同，我们的 Agent 具备以下核心能力：

**（1）显式动作空间。** Agent 不只是在 Prompt 指导下"改进代码"——它从四个离散动作中选择一个：REBUILD（推倒重建骨架）、DELETE（删除有害/冗余组件）、ADD（添加缺失信号类别）、TUNE（微调系数）。每个动作对应不同的 Prompt 策略和代码生成模式。其中 DELETE 是文献中首次被明确定义为奖励搜索的一等动作，解决了迭代搜索中组件持续膨胀的问题。

**（2）结构化 Agent Memory。** 现有 Agent Memory 研究（MIRA、MemRL）将记忆用于 RL 策略学习。我们将 Agent Memory 引入奖励函数搜索领域：存储每轮生成的完整奖励代码、LLM 选择的动作类型、诊断理由和训练结果。Agent Memory 以演化表格（展示每轮的动作-分数-诊断）和全量代码两种形式呈现，支持 Agent 进行跨轮代码对比和证据驱动的决策。诊断理由由 LLM 生成并写入 Memory，形成可追溯的诊断链。

**（3）骨架质量诊断——Agent 的规划能力。** Agent 在每次决策前执行一个结构化的 6 步推理链（Skeleton Quality Diagnosis）：Step 0 判断当前骨架是否值得修复（考察全部历史分数的趋势而非绝对值）→ Step 1 盘点所有组件及其物理信号 → Step 2 识别缺失信号类别 → Step 3 检查信号方向、量级和冗余 → Step 4 做出 REBUILD/DELETE/ADD/TUNE 的决策 → Step 5 生成新奖励函数。这一推理链使 Agent 能够进行组件级的因果分析，而非简单的分数驱动的试错。

**（4）感知-规划-行动闭环。** Agent 以结构化 JSON 格式输出决策（动作类型、操作目标、基于 Memory 证据的诊断理由），代码解析 JSON 并执行对应的操作。训练结果和诊断理由回流至 Agent Memory，形成 `perceive(memory + feedback) → plan(skeleton diagnosis) → act(JSON decision + code) → observe(PPO training) → remember(memory)` 的完整 Agent 循环。

---

## 二、流程详细说明

### 阶段一：环境上下文初始化

系统加载两项 Eureka 式上下文：

1. **任务描述文件**（task_description.txt）：黑盒任务陈述，不包含环境语义
2. **遮罩后的环境 step 源码**（step_masked.py）：官方奖励计算部分被替换为 `<OFFICIAL_REWARD_MASKED>`，仅保留状态向量的构造逻辑（如 `state[0] = pos.x`、`state[4] = lander.angle`、`state[6] = legs[0].ground_contact` 等）

Agent 还接收环境的通用接口描述（观测空间 Box(8,)、动作空间 Discrete(4)）。

### 阶段二：Agent 初始生成（iter0）

Agent 基于初始 Prompt 生成第一个奖励函数 `compute_reward()`。Prompt 包含：
- Agent 角色描述和代码格式约束
- HRDC 结构要求（组件 + training_progress 驱动的阶段权重）
- 禁止使用 `original_reward` 的硬约束
- Eureka 上下文（任务描述 + 遮罩后的 step 源码）

生成的代码经过四层安全校验：AST 语法检查 → 奖励泄露守卫（检测 `original_reward` 引用）→ 领域知识守卫（检测禁止的任务语义术语）→ 运行时烟雾测试（实际执行 3 步验证不报错）。校验失败则反馈错误信息并重试（最多 3 次）。

### 阶段三：PPO 训练与评估

通过 CustomRewardWrapper 将 Agent 生成的奖励函数注入 PPO 训练。训练采用 RL Zoo 对齐的超参数（4 并行环境、1M 步、learning_rate=2.5e-4、n_steps=1024）。训练完成后，在**原始环境奖励**下进行 20 局确定性策略评估，获得平均分数、成功率和平均 episode 长度。

### 阶段四：Agent 反思与决策（迭代核心）

每轮迭代后，Agent 执行完整的感知-规划-行动循环：

**感知（Perceive）**：Agent 接收本轮训练数据（分数、episode 长度、每步奖励、错误计数）、守卫检查状态（original_reward 是否被使用、领域术语是否违规）、以及上轮 LLM 的诊断结论。

**规划（Plan）**：Agent 执行骨架质量诊断 6 步推理链。Step 0 首先判断当前骨架是否值得继续修复——考察全部历史分数的**趋势**而非绝对值。如果分数单调改善（即使从 -400 到 -200），继续调参；如果分数横盘或下降且 episode 长度持续 < 200，判定骨架为"不值得修复"，触发 REBUILD。Steps 1-4 完成组件盘点、缺失识别、方向/量级/冗余检查、最终决策。

**决策（Act）**：Agent 以结构化 JSON 格式输出其决策——选择 REBUILD（推倒重建）、DELETE（删除有害组件）、ADD（添加缺失信号）或 TUNE（微调系数），并附上基于证据的诊断理由。同时生成新的 `compute_reward()` 代码。

**观察（Observe）**：新奖励函数经过安全校验后进入 PPO 训练，Agent 观察训练结果。

**记忆（Remember）**：Agent 将本轮完整决策（动作类型、操作目标、诊断理由、生成的代码、训练分数）存入 Agent Memory。Agent Memory 以演化表格和全量代码两种形式呈现给下一轮决策。

### 阶段五：循环与终止

Agent 在以下条件之一满足时终止搜索：(1) 评估分数达到目标阈值（200 分），(2) 达到最大迭代次数（10 轮），(3) 连续多轮无改善触发 patience 机制。

---

## 三、伪代码

```
算法: FDRE-HRDC Agent Reward Search

输入: 环境E, 最大迭代次数K, 目标分数T
输出: 最优奖励函数 R*

1.  context ← LoadEurekaContext(E)     // 遮罩step源码 + 任务描述
2.  memory ← AgentMemory()             // 空记忆
3.  R_cur, R_best ← None, S_best ← -∞
4.
5.  for iter = 0 to K-1 do
6.      // —— 构建 Prompt ——
7.      if R_cur is None then
8.          prompt ← BuildInitialPrompt(context, HRDC_rules)
9.      else
10.         prompt ← BuildRefinePrompt(
11.             context, R_cur, R_best,
12.             feedback,            // 训练数据 + 规则提醒 + 上轮诊断
13.             memory.Render(),     // Agent Memory（全量代码+演化表）
14.             SkeletonDiagnosis6Steps  // 骨架诊断推理链
15.         )
16.
17.     // —— Agent 生成决策 ——
18.     response ← LLM.Complete(prompt)
19.     (action, target, reasoning, R_new) ← ParseAgentDecision(response)
20.         // action ∈ {REBUILD, DELETE, ADD, TUNE}
21.         // reasoning 由LLM基于Memory中的证据生成
22.
23.     // —— 安全校验 ——
24.     ValidateCode(R_new)           // AST + 泄露 + 领域 + 烟雾测试
25.
26.     // —— PPO 训练 ——
27.     policy ← TrainPPO(E, R_new, 1M steps)
28.     S_eval ← EvaluateOnOriginalReward(E, policy, 20 eps)
29.
30.     // —— 更新状态 ——
31.     if S_eval > S_best then
32.         R_best ← R_new, S_best ← S_eval
33.     R_cur ← R_new
34.
35.     // —— Agent 记忆 ——
36.     memory.Record(iter, action, target, reasoning, S_eval, R_new)
37.
38.     // —— 生成反馈 ——
39.     feedback ← FormatFeedback(S_eval, guard_status, memory.Last().reasoning)
40.
41.     // —— 终止判断 ——
42.     if S_eval ≥ T or PatienceExhausted() then break
43.
44. return R_best
```

---

## 四、模块流程框图

```
┌─────────────────────────────────────────────────────────────────┐
│                    ① Eureka 环境上下文                           │
│  ┌──────────────────────┐  ┌──────────────────────────────────┐ │
│  │ task_description.txt │  │ step_masked.py                   │ │
│  │ "Black-box control   │  │ state[0]=pos.x  state[4]=angle   │ │
│  │  task. Maximize      │  │ state[6]=leg_contact             │ │
│  │  original return."   │  │ reward = <OFFICIAL_MASKED>       │ │
│  └──────────────────────┘  └──────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│              ② Agent Prompt 构建（含 ③ HRDC 结构约束）           │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ System: 你是自主奖励设计Agent。每个组件必须有基于             │ │
│  │ training_progress 的独立阶段权重（HRDC）。                    │ │
│  │ 官方奖励已遮罩，禁止使用 original_reward。                   │ │
│  │ 输出格式: JSON决策 + Python代码                              │ │
│  └────────────────────────────────────────────────────────────┘ │
│  ┌────────────────────┐ ┌──────────────┐ ┌──────────────────┐  │
│  │ ⑥ Agent Memory     │ │ 当前反馈     │ │ ④⑤ 骨架诊断     │  │
│  │ (全量代码+演化表)   │ │ (数据+诊断)  │ │ (6步推理链)      │  │
│  └────────────────────┘ └──────────────┘ └──────────────────┘  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                  ③ Agent 决策（LLM 推理）                        │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ { "action": "delete",                                      │ │
│  │   "target": "height_penalty",                              │ │
│  │   "reasoning": "iter1(无height)=109 > iter2(有height)=-122"│ │
│  │ }                                                          │ │
│  │ def compute_reward(...): ...                               │ │
│  └────────────────────────────────────────────────────────────┘ │
│  动作空间: REBUILD | DELETE | ADD | TUNE                        │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ④ 四层安全校验                                │
│  AST语法 → 奖励泄露守卫 → 领域知识守卫 → 运行时烟雾测试          │
└──────────────────────────┬──────────────────────────────────────┘
                           │ 通过
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                 ⑤ PPO 训练 + 原始环境评估                        │
│  CustomRewardWrapper(R_new) → PPO(1M steps) → 20局确定性评估    │
│  → 分数 / episode长度 / 成功率                                  │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
┌─────────────────────────────────────────────────────────────────┐
│                    ⑥ Agent Memory 更新                          │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Evolution Summary:                                         │ │
│  │ | iter | action | target     | score | diagnosis (LLM)    │ │
│  │ | 0    | init   | skeleton   | -242  | first attempt...  │ │
│  │ | 1    | add    | contact    | -127  | died early, add.. │ │
│  │ | 2    | delete | height     |  109  | height harmful.. │ │
│  │                                                            │ │
│  │ ## Iteration 0 (全量代码)                                   │ │
│  │ ## Iteration 1 (全量代码)                                   │ │
│  └────────────────────────────────────────────────────────────┘ │
└──────────────────────────┬──────────────────────────────────────┘
                           │
                           ▼
              ┌────────────────────────┐
              │ 终止? S_best ≥ 200 或  │
              │ patience 或 max_iter   │
              └───────────┬────────────┘
                    Yes ↓         No → 回到②
              ┌────────────┐
              │ 返回 R_best │
              └────────────┘
```

## 五、创新点在图中的位置

| 编号 | 创新点 | 在流程图中位置 |
|------|--------|---------------|
| ① | 遮罩式无先验搜索 | 模块① — Eureka 上下文 |
| ② | HRDC 阶段动态权重 | 模块② — Prompt 中的结构约束 |
| ③ | 骨架质量诊断 6 步 | 模块② — 注入 Prompt 的推理链 |
| ④ | Agent 决策 JSON + 动作空间 | 模块③ — LLM 输出结构化决策 |
| ⑤ | DELETE 作为一等动作 | 模块③ — 动作空间 |
| ⑥ | 代码级 Agent Memory | 模块⑥ — 全量代码 + 演化表 + LLM 诊断链 |
