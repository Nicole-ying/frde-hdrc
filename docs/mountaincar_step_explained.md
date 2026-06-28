# MountainCar step 函数完整解读

## 环境概述

MountainCar-v0：一辆小车停在山谷底部，目标是通过来回摆动冲上右侧山顶。发动机太弱，一次加速上不去——必须借助重力势能来回蓄力。

- **动作空间**：Discrete(3) 离散
- **观测空间**：2 维（位置、速度）

---

## step 函数完整注释

```python
def step(self, action):
    # action ∈ {0, 1, 2}：
    #   0 = 向左加速（推左）
    #   1 = 不加速（滑行）
    #   2 = 向右加速（推右）

    position, velocity = self.state  # 当前状态：位置和速度

    # ====== 第一步：更新速度 ======
    # 加速度 = 引擎推力 + 重力分量
    #   引擎推力：(action - 1) × force
    #     action=0 → (0-1)×0.001 = -0.001（向左加速）
    #     action=1 → (1-1)×0.001 =  0     （不加速）
    #     action=2 → (2-1)×0.001 = +0.001（向右加速）
    #
    #   重力分量：cos(3×position) × (-gravity)
    #     地形是 cos(3x) 形状的山谷——位置不同，坡度不同
    #     gravity = 0.0025
    #     在谷底（position≈-0.5）：cos(-1.5)≈0.07 → 重力≈-0.00018（几乎平的）
    #     在山坡（position≈-1.2）：cos(-3.6)≈-0.9 → 重力≈+0.00225（滑向右）
    #     在山坡（position≈0.2）：cos(0.6)≈0.83 → 重力≈-0.0021（滑向左）

    velocity += (action - 1) * self.force + cos(3 * position) * (-self.gravity)

    # ====== 第二步：限制速度范围 ======
    velocity = clip(velocity, -0.07, 0.07)  # 最大速度 ±0.07

    # ====== 第三步：更新位置 ======
    position += velocity
    position = clip(position, -1.2, 0.6)  # 位置范围 [-1.2, 0.6]

    # ====== 第四步：左墙弹性碰撞 ======
    # 撞到最左边且速度向左 → 速度归零（弹性碰撞）
    if position == -1.2 and velocity < 0:
        velocity = 0

    # ====== 第五步：成功判断 ======
    # 到达右山顶（position >= 0.5 且速度 >= 0）
    terminated = (position >= 0.5 and velocity >= 0)

    # ====== 第六步：奖励 ======
    reward = -1.0  # 每步扣 1 分！逼 agent 尽快到达终点

    self.state = (position, velocity)
    return [position, velocity], reward, terminated, False, {}
```

## 奖励函数的数学本质

```
reward = -1.0（每一步）

总奖励 = -总步数
```

**极简设计**：没有 shaping，没有增量奖励。每活一步扣 1 分，越快到终点扣分越少。最优策略是尽快上山——但因为引擎太弱，必须来回摆动蓄力。

## 为什么 MountainCar 的奖励这么简单？

因为它不需要 shaping：
- LunarLander/BipedalWalker 是连续控制，shaping 提供稠密的学习信号
- MountainCar 只有 2 维状态，Sarsa/Q-learning 就能解
- 这个环境考验的是"探索还是利用"——agent 必须学会先远离目标（向左开蓄力）再冲上去

## 三环境奖励对比

| | LunarLander | BipedalWalker | MountainCar |
|---|---|---|---|
| **观测维度** | 8 | 24 | 2 |
| **动作空间** | Discrete(4) | Box(4) | Discrete(3) |
| **shaping** | 距离+速度+角度+触地 | 前进距离+头部直立 | 无 |
| **增量式** | ✅ shaping改善量 | ✅ shaping改善量 | ❌ 固定-1/步 |
| **燃料惩罚** | ✅ m_power/s_power | ✅ Σ|action_i| | 无 |
| **终止奖励** | 坠毁-100 着陆+100 | 摔倒-100 | 无 |
| **目标** | 平稳降落 | 走得远 | 冲到山顶 |

## 我们的签名在这个环境下的适用性

```
compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0)
```

- `obs`/`next_obs`：2 维 [position, velocity]，双帧可计算"速度变化量"
- `action`：0/1/2 离散
- `original_reward`：永远是 -1.0（没用）
- `training_progress`：HRDC 可用——早期鼓励探索（离开谷底），晚期鼓励冲刺
