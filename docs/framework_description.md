# FDRE-HRDC 框架论文说明

## 创新点

### 创新一：HRDC — 层次化奖励动态组合

传统 LLM 奖励生成方法（EUREKA、Text2Reward、CARD）允许 LLM 自由设计权重结构，导致同一 prompt 下生成质量高度随机——LLM 可能输出两阶段插值、九维连续权重或三角窗归一化。

HRDC（Hierarchical Reward Decomposition with Dynamic Weights）强制要求每个奖励组件拥有随 training_progress 演化的独立阶段权重，形成"早期探索→中期平衡→晚期精度"的结构先验。这一约束作为 Prompt 中的硬性要求传递给 LLM，将权重设计从自由创作转变为结构化搜索，显著降低搜索方差。

### 创新二：Autonomous Reward Design Agent

我们将奖励函数搜索重新定义为 Agent 决策问题。与现有的"Prompt→代码→训练→反馈"管道不同，我们的 Agent 具备三个核心能力：

**Agent Memory（代码级记忆）。** 存储每轮生成的完整奖励代码、LLM 选择的动作类型、诊断理由和训练分数，以演化表格和全量代码两种形式呈现。与 MIRA、MemRL 将记忆用于策略学习不同，我们的 Agent Memory 专用于奖励函数的跨轮代码对比和证据驱动决策。

**显式动作空间。** Agent 从 {REBUILD, DELETE, ADD, TUNE} 四个离散动作中自主选择，以结构化 JSON 格式输出决策。其中 DELETE 首次将组件删除定义为奖励搜索的一等动作，解决了迭代搜索中组件持续膨胀的问题。

**骨架质量诊断（规划能力）。** Agent 在每次决策前执行 6 步推理链：Step 0 考察全部历史趋势判断骨架是否值得修复（< 3 轮禁止重建）→ Step 1 盘点组件 → Step 2 识别缺失信号 → Step 3 检查方向/量级/冗余 → Step 4 做出动作决策 → Step 5 生成代码。

---

## 实验结果

**v2 实验（3 seed，骨架诊断验证）。** 三个种子全部从负分起点（-207、-140、-116）通过迭代搜索达到 solved 标准，最终分数分别为 226、233、211。详见 `figures/A1-A4`。

**10-seed 实验（大规模验证）。** 十个完全独立的种子中，5 个成功搜索到 solved 级别的奖励函数（成功率 50%），最高分 253。详见 `figures/B1-B6`。

**证据链。** PPO baseline（官方奖励，260 分）→ v2 3-seed（223 分均值）→ 10-seed solved 均值，证明 Agent 生成的奖励函数可匹敌官方奖励。详见 `figures/C1-C2`。

---

## 流程说明

### 阶段一：环境上下文

系统加载 Eureka 式上下文：任务描述文件（黑盒陈述）和遮罩后的 step 源码（官方奖励替换为 `<OFFICIAL_REWARD_MASKED>`，仅保留状态向量构造逻辑）。Agent 接收观测空间 Box(8,) 和动作空间 Discrete(4) 的通用接口描述。

### 阶段二：初始生成（iter0）

Agent 基于初始 Prompt（角色描述 + HRDC 结构约束 + Eureka 上下文）生成第一个 `compute_reward()`。代码经四层校验：AST 语法 → 奖励泄露守卫（禁止 `original_reward`）→ 领域知识守卫 → 运行时烟雾测试。校验失败则反馈错误并重试。

### 阶段三：PPO 训练与评估

Agent 生成的奖励通过 CustomRewardWrapper 注入 PPO 训练（4 并行环境、1M 步、learning_rate=2.5e-4）。训练后在原始环境奖励下进行 20 局确定性评估，获得分数、成功率和 episode 长度。

### 阶段四：Agent 循环（迭代核心）

**感知：** Agent 接收训练数据（分数、episode 长度、每步奖励）、守卫状态、上轮 LLM 诊断。

**规划：** 执行骨架质量诊断 6 步推理链，考察全部历史趋势而非单一分数。

**决策：** 以 JSON 输出动作（REBUILD/DELETE/ADD/TUNE）+ 诊断理由 + 新代码。

**观察：** 新奖励经安全校验后进入 PPO 训练，Agent 获取结果。

**记忆：** 完整决策存入 Agent Memory，以演化表格和全量代码呈现给下一轮。

### 阶段五：终止

满足任一条件终止：分数 ≥ 200、达到最大迭代次数、连续多轮无改善。

---

## 伪代码

```
算法: FDRE-HRDC Agent Reward Search
输入: 环境E, 最大迭代K, 目标分数T
输出: 最优奖励函数 R*

1.  context ← LoadEurekaContext(E)
2.  memory ← AgentMemory()
3.  R_best ← None, S_best ← -∞

4.  for iter = 0 to K-1 do
5.      if R_cur is None:
6.          prompt ← BuildInitialPrompt(context, HRDC_rules)
7.      else:
8.          prompt ← BuildRefinePrompt(context, R_cur, R_best,
9.                      feedback, memory.Render(), SkeletonDiagnosis)
10.
11.     response ← LLM.Complete(prompt)
12.     (action, target, reasoning, R_new) ← ParseAgentDecision(response)
13.     ValidateCode(R_new)
14.
15.     policy ← TrainPPO(E, R_new)
16.     S_eval ← Evaluate(E, policy)
17.
18.     if S_eval > S_best: R_best ← R_new, S_best ← S_eval
19.     memory.Record(iter, action, target, reasoning, S_eval, R_new)
20.     feedback ← Format(S_eval, guard_status, memory.Last().reasoning)
21.
22.     if S_eval ≥ T or Patience(): break

23. return R_best
```

---

## 模块流程框图

```
┌─────────────────────────────────────────────────────┐
│               Eureka 环境上下文                       │
│  task.txt (黑盒描述) + step_masked.py (遮罩源码)      │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│           ② Agent Prompt 构建                        │
│  ┌──────────────────────────────────────────────┐   │
│  │ HRDC 结构约束 + 禁止 original_reward          │   │
│  │ 输出格式: JSON决策 + Python代码               │   │
│  └──────────────────────────────────────────────┘   │
│  ┌──────────────┐ ┌──────────┐ ┌───────────────┐   │
│  │ Agent Memory │ │ Feedback │ │ 骨架诊断(6步)  │   │
│  │ (全量代码    │ │ (数据    │ │ (规划能力)     │   │
│  │  +演化表)    │ │  +诊断)  │ │               │   │
│  └──────────────┘ └──────────┘ └───────────────┘   │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              ③ Agent 决策 (LLM)                     │
│  { "action": "delete", "target": "height_penalty",  │
│    "reasoning": "证据..." }                          │
│  def compute_reward(...): ...                        │
│  动作: REBUILD | DELETE | ADD | TUNE               │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              ④ 安全校验 + PPO训练 + 评估             │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────┐
│              ⑤ Agent Memory 更新                    │
│  | iter | action | target | score | LLM诊断 |       │
│  + 每轮全量奖励代码                                   │
└────────────────────────┬────────────────────────────┘
                         │
                         ▼
              ┌──────────────────┐
              │ 终止判断 → 返回   │
              └──────────────────┘
```

### 两个创新点在图中位置

| 创新点 | 位置 |
|--------|------|
| **HRDC** | 模块② — Prompt 中的结构约束，贯穿整个生成过程 |
| **Agent** | 模块②(记忆+规划) → ③(决策+动作空间) → ⑤(记忆更新)，形成完整的感知-规划-行动闭环 |
