# LunarLander step 函数解读 + 参数对比

## 整体流程（一句话）

每帧物理模拟 → 根据 action 施加推力 → 更新位置/速度/角度 → 组装 8 维观测向量 → 计算奖励

---

## step 函数逐段注释

```python
def step(self, action):
    # ========== 第一部分：风力扰动（可忽略） ==========
    # 落地前有随机风力和扭矩，影响飞行姿态
    # 腿着地后风停

    # ========== 第二部分：处理动作 ==========
    # action 是 0/1/2/3 四个离散值：
    #   0 = 什么都不做
    #   1 = 点燃左方向引擎（向左转）
    #   2 = 点燃主引擎（向下喷，产生升力）
    #   3 = 点燃右方向引擎（向右转）

    # ========== 第三部分：施加物理力 ==========
    # 主引擎（action=2）：
    #   → 沿着机身 tip 方向施加推力
    #   → m_power = 1.0（离散动作，满油门）
    #
    # 方向引擎（action=1 或 action=3）：
    #   → 沿着机身 side 方向施加推力
    #   → s_power = 1.0

    # ========== 第四部分：物理模拟一步 ==========
    self.world.Step(...)  # Box2D 物理引擎推进一帧

    # ========== 第五部分：组装观测向量（8 维） ==========
    pos = self.lander.position   # 着陆器的 (x, y) 坐标
    vel = self.lander.linearVelocity  # 着陆器的 (vx, vy) 速度

    state = [
        state[0]: 水平位置 x（归一化到 [-1, 1]），0=中心
        state[1]: 垂直位置 y（归一化到 [-1, 1]），负=接近地面
        state[2]: 水平速度 vx（归一化）
        state[3]: 垂直速度 vy（归一化）
        state[4]: 机身倾斜角度 angle（弧度），0=直立
        state[5]: 角速度 angularVelocity，0=不旋转
        state[6]: 左腿是否触地（0 或 1）
        state[7]: 右腿是否触地（0 或 1）
    ]

    # ========== 第六部分：终止判断 ==========
    terminated = True 如果：
      - 坠毁（game_over）
      - 飞出屏幕（|x| > 1）
      - 着陆器不再活跃（lander.awake = False）

    # ========== 第七部分：计算奖励 ==========
    # 🎯 这一行就是 LLM 要生成的 compute_reward 函数：
    reward, components = self.compute_reward(state, m_power, s_power, terminated)
    #
    # 传入的参数：
    #   state:    8 维观测向量（位置/速度/角度/触地）
    #   m_power:  主引擎推力大小（0.0 或 1.0）
    #   s_power:  方向引擎推力大小（0.0 或 1.0）
    #   terminated: 是否已终止（True/False）
    #
    # 返回值：
    #   reward:    总奖励（浮点数）
    #   components: 字典，如 {"stability": 0.5, "contact": 1.0, ...}

    # 📊 这一行是官方的"评估指标"（不用于训练，只用于比较）：
    fitness_score = self.compute_fitness_score(state, m_power, s_power, terminated)
```

---

## 三方参数对比

| | 官方 Gymnasium | Eureka（LLM 生成） | 我们的框架 |
|---|---|---|---|
| **调用方式** | `self.compute_reward(state, m_power, s_power, terminated)` | 同官方 | `compute_reward(obs, action, next_obs, original_reward, info, training_progress)` |
| **参数 1** | `state` — 8 维归一化向量 | 同官方 | `obs` — 当前 8 维向量 |
| **参数 2** | `m_power` — 主引擎推力（0 或 1） | 同官方 | `action` — 动作编号 0/1/2/3 |
| **参数 3** | `s_power` — 方向引擎推力（0 或 1） | 同官方 | `next_obs` — 下一步的 8 维向量 |
| **参数 4** | `terminated` — 是否结束（True/False） | 同官方 | `original_reward` — 环境原始奖励 |
| **参数 5** | — | — | `info` — 环境额外信息字典 |
| **参数 6** | — | — | `training_progress` — 0 到 1，训练进度 |
| **返回值** | `(float, dict)` | `(float, dict)` | `float` |

## 为什么不一样？

官方/Eureka 的 `compute_reward` 是**环境类内部方法**，在 `step()` 函数里被调用。它自然能访问 `state`、`m_power`、`s_power`、`terminated`——这些都是 step 函数内部的局部变量。

我们的 `compute_reward` 是**独立函数**，通过 `CustomRewardWrapper` 在外部调用。Wrapper 只知道 Gymnasium 的标准接口：
- 上一帧的观测 `obs`
- 当前动作 `action`
- 下一帧的观测 `next_obs`
- 环境原始奖励 `original_reward`
- 环境信息字典 `info`

所以我们给了更通用的参数，而不是环境特定的 `m_power`/`s_power`/`terminated`。

### 参数含义对照

| 我们的参数 | 实际含义 | 等效于官方的... |
|-----------|---------|---------------|
| `obs` | 当前 8 维状态 | `state`（上一帧） |
| `action` | 0=无 1=左 2=主 3=右 | 可以推导 `m_power` 和 `s_power` |
| `next_obs` | 下一帧 8 维状态 | `state`（当前帧） |
| `original_reward` | 官方奖励值 | 同官方 `reward` |
| `info` | 额外信息（通常为空） | — |
| `training_progress` | 0→1，训练完成比例 | 🆕 我们的创新（HRDC 用） |

### 我们用 `obs`+`next_obs` 而不是 `state` 的原因

`obs` 和 `next_obs` 两个相邻帧让我们能计算**转移信号**——速度变化、角度变化、是否在靠近目标等。这是官方 `state` 做不到的（只有一个时间点）。

### `training_progress` 是独有创新

官方和 Eureka 都没有这个参数。它让奖励函数知道"训练进行到哪了"，从而实现 HRDC——早期探索、晚期精度的阶段权重。
