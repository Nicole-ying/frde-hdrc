# LunarLander step 函数完整解读 + 三方法签名对比

## 官方 reward 计算逻辑

```python
def step(self, action):
    # ====== 第一部分：风力扰动（着陆前有随机风） ======
    # 落地前有随机风力和扭矩，影响飞行姿态。忽略即可。

    # ====== 第二部分：处理动作 ======
    # action ∈ {0,1,2,3} 离散动作：
    #   0 = 什么都不做（滑行）
    #   1 = 点燃左方向引擎（机身向左转）
    #   2 = 点燃主引擎（向下喷气，产生向上推力）
    #   3 = 点燃右方向引擎（机身向右转）

    # ====== 第三部分：施加物理力 ======
    # 主引擎（action=2）：m_power=1.0，沿机身 tip 方向喷气
    # 方向引擎（action=1/3）：s_power=1.0，沿机身 side 方向喷气
    # (物理公式不需要理解，知道它们在模拟推力即可)

    # ====== 第四部分：Box2D 物理引擎推进一帧 ======
    self.world.Step(1.0/FPS, 6*30, 2*30)

    # ====== 第五部分：组装 8 维观测向量 ======
    pos = self.lander.position        # (x, y) 着陆器坐标
    vel = self.lander.linearVelocity  # (vx, vy) 速度

    state = [
        state[0]: 水平位置 x（归一化到 [-1,1]），0=正中间
        state[1]: 垂直位置 y（归一化到 [-1,1]），负值=接近地面
        state[2]: 水平速度 vx（归一化）
        state[3]: 垂直速度 vy（归一化）
        state[4]: 机身倾斜角 angle（弧度），0=完全直立
        state[5]: 角速度（旋转速度），0=不转
        state[6]: 左腿触地标志（0=空中, 1=触地）
        state[7]: 右腿触地标志（0=空中, 1=触地）
    ]

    # ====== 第六部分：计算奖励（🔥 这就是 LLM 要生成的函数） ======
    reward = 0

    # 步骤A：计算 shaping 值（衡量"当前状态有多好"）
    shaping = (
        -100 * sqrt(x^2 + y^2)      # 离着陆点越远，扣分越多
        -100 * sqrt(vx^2 + vy^2)    # 速度越快，扣分越多（防坠毁）
        -100 * abs(angle)            # 越倾斜，扣分越多（防翻跟头）
        +10  * legs[0]               # 左腿触地 +10分
        +10  * legs[1]               # 右腿触地 +10分
    )

    # 步骤B：奖励 = shaping 的改善量（增量式奖励）
    # 如果上一帧 shaping=-200，这一帧 shaping=-100
    # → reward = (-100) - (-200) = +100（改善了很多！）
    reward = shaping - prev_shaping

    # 步骤C：扣除燃料消耗
    reward -= m_power * 0.30   # 主引擎推力越大，扣分越多
    reward -= s_power * 0.03   # 方向引擎推力越大，扣分越多

    # 步骤D：终止事件的奖惩
    if game_over（坠毁）:     reward = -100  # 坠毁罚100分
    if not awake（成功着陆）:  reward = +100  # 成功着陆奖100分
```

### 官方奖励函数的数学本质

```
总奖励 = 本次 shaping - 上次 shaping   ← 增量式：鼓励状态改善
        - 0.30 × m_power              ← 惩罚燃料消耗
        - 0.03 × s_power
        ← 坠毁则 -100，成功着陆则 +100
```

**核心思想**：不是"状态好给高分"，而是"状态在改善给高分"（potential-based reward shaping）。

---

## 三方法 compute_reward 签名对比

| | 官方 Gymnasium | Eureka（LLM生成） | 我们 FDRE-HRDC |
|---|---|---|---|
| **调用位置** | step() 内部 | step() 内部 | CustomRewardWrapper 外部 |
| **签名** | 无独立函数，reward 计算内联在 step() 中 | `def compute_reward(self, ...): return reward, {}` | `def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0): return float` |
| **参数1** | — | `self` — env 实例 | `obs` — 当前 8 维向量 |
| **参数2** | — | LLM 自选（通常从 step 看到 state/m_power/s_power/terminated） | `action` — 动作 0/1/2/3 |
| **参数3** | — | — | `next_obs` — 下一帧 8 维向量 |
| **参数4** | — | — | `original_reward` — 官方奖励 |
| **参数5** | — | — | `info` — 环境信息字典 |
| **参数6** | — | — | 🆕 `training_progress` — 训练进度 0→1 |
| **返回值** | float（内联） | `(float, dict)` | `float` |

## 为什么我们不一样

### 1. 为什么用 `obs`+`next_obs` 而不是 `state`？

官方和 Eureka 只有一个 `state`（当前帧）。我们给 `obs`+`next_obs` 两个相邻帧，让奖励函数能计算**转移信号**：
- `abs(obs) - abs(next_obs)` → 状态在收敛还是发散？
- `next_obs[4] - obs[4]` → 角度在变大还是变小？
- `next_obs[6]+next_obs[7]` → 是否刚触地？

这比单帧信息更丰富。LLM 能从相邻帧的变化中提取学习信号。

### 2. 为什么传 `action` 而不是 `m_power`/`s_power`？

`action` 是 Gymnasium 的标准接口，任何环境都有。`m_power`/`s_power` 是 LunarLander 特有的内部变量。用 `action` 让奖励函数更通用——LLM 能从 `action` 推导引擎使用情况（action=2 → 主引擎，action=1/3 → 方向引擎）。

### 3. `training_progress` — 我们的独有创新

官方和 Eureka 都没有这个参数。它让奖励函数感知训练阶段，实现 HRDC：

```
早期 (progress<0.3):  惩罚要轻，让 agent 自由探索
中期 (0.3~0.7):      探索和精度平衡
晚期 (progress>=0.7): 精度优先，加强接触和稳定性信号
```

### 4. 为什么返回值只有 float 而不是 (float, dict)？

简单性。返回组件字典需要 LLM 额外处理，增加出错概率。Eureka 用组件字典是为了 reflection（追踪每个组件的 max/mean/min）。我们可以后续添加组件追踪而不改签名。
