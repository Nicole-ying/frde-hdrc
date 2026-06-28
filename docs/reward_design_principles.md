# 奖励函数设计范式与参数选择

## 一、我们的参数为什么合理且通用

### 固定签名的来源

```python
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    return total_reward
```

这 6 个参数全部来自 Gymnasium 标准接口——每一帧 `env.step(action)` 都返回这些：

```python
next_obs, original_reward, terminated, truncated, info = env.step(action)
```

| 参数 | 来源 | 为什么有用 | 适用性 |
|------|------|-----------|--------|
| `obs` | step 前的观测 | 当前状态 | ✅ 所有环境 |
| `action` | 刚刚执行的动作 | 动作代价（燃料/力矩） | ✅ 所有环境 |
| `next_obs` | step 后的观测 | 转移信号（状态改善还是恶化？） | ✅ 所有环境 |
| `original_reward` | 环境内置奖励 | 可作为锚点或参考 | ✅ 所有环境 |
| `info` | 环境字典 | 额外信息（如触地标志） | ✅ 所有环境 |
| `training_progress` | 框架注入（0→1） | HRDC 阶段权重 | ✅ 我们的创新 |

### 与其他方法的通用性对比

| | Eureka | 我们的 |
|---|---|---|
| **LunarLander** | `(state, m_power, s_power, terminated)` | `(obs, action, next_obs, ...)` ✅ |
| **BipedalWalker** | 需重新适配参数 | `(obs, action, next_obs, ...)` ✅ |
| **MountainCar** | 需重新适配参数 | `(obs, action, next_obs, ...)` ✅ |
| **任意 Gym 环境** | 需修改 step 源码 | 开箱即用 ✅ |

Eureka 的参数是**环境专属**的——LLM 看到 step 源码后自己写参数名。我们固定了**通用的 Gymnasium 接口参数**——不需要改环境代码，任意环境都能用。

### `obs`+`next_obs` 双帧设计的价值

单帧只能看"现在什么样"，双帧能看"变化方向"：

```
obs=[0.5, -0.3, ...]      ← 位置偏右，接近地面
next_obs=[0.3, -0.5, ...]  ← 位置趋于中心，更接近地面

→ abs(obs[0])-abs(next_obs[0]) = 0.5-0.3 = +0.2  奖励：在靠近中心！
→ abs(obs[1])-abs(next_obs[1]) = 0.3-0.5 = -0.2  惩罚：下降太快！
```

这是官方 `state` 单参数做不到的——它是增量式奖励的基础。

---

## 二、奖励函数设计四大范式

### 范式 A：稀疏/终点奖励

```
只有任务完成或失败时才给奖励。
例：MountainCar 的 reward = -1/step（稀疏但每步都有）
    棋盘游戏：赢了+1，输了-1
    抓取任务：抓住了+1，没抓住0
```

**优点**：没有人为偏置，学到的是最优策略
**缺点**：学习信号极少，难以收敛（需要大量探索）

### 范式 B：密集/距离奖励

```
每一步都给奖励，基于离目标的距离。
例：reward = -distance(agent_position, goal_position)
    离目标越近，扣分越少
```

**优点**：每步都有学习信号，收敛快
**缺点**：可能学到"只靠近目标但不完成任务"的局部最优

### 范式 C：增量式奖励（Potential-Based Shaping）

```
不是"状态好就给高分"，而是"状态改善就给高分"。
reward = shaping(新状态) - shaping(旧状态)
         ↑ 一个衡量"当前状态有多好"的函数

例：LunarLander 的 shaping = -100*距离 -100*速度 -100*倾斜 +10*触地
     reward = shaping_new - shaping_old（鼓励每一步都在改善）

BipedalWalker 的 shaping = 130*前进距离 - 5*头部倾斜
reward = shaping_new - shaping_old
```

**优点**：理论保证不改变最优策略（policy invariance），同时提供稠密信号
**缺点**：需要设计一个好的 shaping 函数

### 范式 D：组件式奖励（Component-Based Reward）

```
reward = w1 × 组件1 + w2 × 组件2 + w3 × 组件3 + ...

每个组件衡量任务的一个方面：
  组件1: 稳定性（角度惩罚）
  组件2: 进度（接近目标）
  组件3: 效率（低能耗）
  组件4: 安全性（不坠毁）
  ...

这是我们的框架和 Eureka 都生成的类型。
HRDC 让权重 w1...wn 随 training_progress 动态变化。
```

**优点**：可解释、可分解、可独立调优
**缺点**：权重难以平衡，组件之间可能冲突

---

## 三、我们的奖励类型 vs Eureka

| | Eureka | 我们 |
|---|---|---|
| **范式** | 组件式 | 组件式 + HRDC 动态权重 |
| **组件数量** | LLM 自由决定 | LLM 自由决定 |
| **权重方式** | 固定权重 | training_progress 驱动的阶段权重 |
| **进化策略** | K 个候选，选最佳 | 单样本顺序迭代 |
| **组件追踪** | 返回 dict，追踪 max/mean/min | 尚未实现（可加） |
| **反思机制** | 数值驱动的 reflection | 文本驱动 + Agent Memory 跨轮对比 |

---

## 四、奖励函数设计原则

### 1. 简单优先（Start Simple）

```
5 个正确的组件 > 10 个微妙的组件
先加最重要的信号（生存/稳定性），再补辅助信号
```

### 2. 方向性（Directional Signal）

```
✅ abs(obs) - abs(next_obs)  → 有方向：在靠近零还是在远离？
❌ (obs - next_obs)^2         → 无方向：任何变化都给正分
```

### 3. 量级平衡（Scale Balance）

```
如果 survival_bonus = 10/步，contact_bonus = 0.1/步
→ survival 淹没了 contact，agent 学会"活着"但不会"降落"

每个组件的每步贡献应该在同一数量级（0.1~2.0）
```

### 4. 阶段适配（Stage-Appropriate）→ HRDC 的核心

```
早期训练 (progress<0.3)：
  惩罚要轻 → agent 需要自由探索
  探索信号权重高 → 鼓励尝试不同策略

晚期训练 (progress>0.7)：
  精度信号权重高 → 接触、稳定、节能
  探索信号权重低 → 不再需要乱试
```

### 5. 避免奖励黑客（Avoid Reward Hacking）

```
坏的设计：reward = +1 每步存活
→ agent 学会原地不动永远活着

好的设计：reward = 生存 + 进度 - 成本
→ agent 必须在活着的同时向目标前进
```

### 6. 增量优于绝对（Delta over Absolute）

```
❌ reward = -distance_to_goal    （绝对距离）
    → agent 从出生就减 100 分，不知道"改善"是什么

✅ reward = -(distance_now - distance_before)（距离变化）
    → agent 每次靠近目标都得正分，每次远离都得负分
```

---

## 五、我们的参数是通用且合理的

1. **所有 Gymnasium 环境都提供** `obs`、`action`、`next_obs`、`reward`、`info`
2. **`obs`+`next_obs` 双帧**天然支持增量式/转移信号设计
3. **`training_progress`** 是独有创新，支持 HRDC 阶段权重
4. 不需要修改环境源码——Eureka 需要

这是从"算法"到"框架"的关键一步：固定签名 = 通用性。
