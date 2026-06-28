# BipedalWalker step 函数完整解读

## 环境概述

BipedalWalker-v3：一个两足行走机器人，需要通过控制四个关节电机在崎岖地形上前进。目标是走得越远越好，同时保持平稳不摔倒。

- **动作空间**：4 维连续向量，每个值 ∈ [-1, 1]
- **观测空间**：24 维向量（机体角度+速度 + 4个关节的角度+速度 + 2个触地标志 + 10个雷达距离）

---

## step 函数完整注释

```python
def step(self, action):
    # action 是一个 4 维向量 [a0, a1, a2, a3]：
    #   a0: 左髋关节 (Left Hip)   — 控制左大腿的前后摆动
    #   a1: 左膝关节 (Left Knee)   — 控制左小腿的弯曲
    #   a2: 右髋关节 (Right Hip)   — 控制右大腿的前后摆动
    #   a3: 右膝关节 (Right Knee)   — 控制右小腿的弯曲
    #   每个值 ∈ [-1, 1]：正值=向前转，负值=向后转，绝对值=力矩大小

    # ====== 第一部分：控制关节电机 ======
    # 每个关节有两个控制量：
    #   motorSpeed：    目标转速（正值=向前转，负值=向后转）
    #   maxMotorTorque：最大力矩（|action|越大，力矩越大）
    #
    # 左腿：joints[0]=髋, joints[1]=膝
    # 右腿：joints[2]=髋, joints[3]=膝
    #
    # 扭矩换算：MOTORS_TORQUE × |action|（action 的绝对值决定力量大小）
    # 转速换算：SPEED_HIP × sign(action) 或 SPEED_KNEE × sign(action)

    self.joints[0].motorSpeed = SPEED_HIP × sign(a0)     # 左髋：方向=sign(a0)，速度=SPEED_HIP
    self.joints[0].maxMotorTorque = MOTORS_TORQUE × |a0| # 左髋：力矩=|a0|
    self.joints[1].motorSpeed = SPEED_KNEE × sign(a1)    # 左膝
    self.joints[1].maxMotorTorque = MOTORS_TORQUE × |a1|
    self.joints[2].motorSpeed = SPEED_HIP × sign(a2)     # 右髋
    self.joints[2].maxMotorTorque = MOTORS_TORQUE × |a2|
    self.joints[3].motorSpeed = SPEED_KNEE × sign(a3)    # 右膝
    self.joints[3].maxMotorTorque = MOTORS_TORQUE × |a3|

    # ====== 第二部分：物理模拟一步 ======
    self.world.Step(1.0/FPS, 6*30, 2*30)

    # ====== 第三部分：激光雷达扫描（10条射线） ======
    # 从机器人中心向下方扇形发射 10 条射线，测量到地面的距离
    # lidar[i].fraction ∈ [0,1]：0=紧贴地面，1=雷达范围内无地面
    # 这 10 个值让机器人"感知"前方地形

    # ====== 第四部分：组装 24 维观测向量 ======
    pos = self.hull.position        # 机器人中心 (x, y) 坐标
    vel = self.hull.linearVelocity  # 机器人中心 (vx, vy) 速度

    state = [
        state[0]:  hull.angle             — 机身倾斜角（弧度），0=直立
        state[1]:  hull.angularVelocity   — 机身角速度（归一化）
        state[2]:  vel.x                  — 水平速度（归一化到 [-1,1]）
        state[3]:  vel.y                  — 垂直速度（归一化到 [-1,1]）
        state[4]:  joints[0].angle        — 左髋关节角度
        state[5]:  joints[0].speed        — 左髋关节转速（归一化）
        state[6]:  joints[1].angle + 1.0  — 左膝关节角度（偏移+1）
        state[7]:  joints[1].speed        — 左膝关节转速（归一化）
        state[8]:  legs[1].ground_contact — 左腿触地（0或1）
        state[9]:  joints[2].angle        — 右髋关节角度
        state[10]: joints[2].speed        — 右髋关节转速（归一化）
        state[11]: joints[3].angle + 1.0  — 右膝关节角度（偏移+1）
        state[12]: joints[3].speed        — 右膝关节转速（归一化）
        state[13]: legs[3].ground_contact — 右腿触地（0或1）
        state[14..23]: lidar[0..9]        — 10条雷达射线距离（0~1）
    ]
    # 总共 14 + 10 = 24 维

    # ====== 第五部分：计算奖励（🔥 核心逻辑） ======

    # 步骤A：shaping 值 — "当前状态有多好"
    shaping = 0

    # 奖励向前移动（每走远一点，奖励就增加）
    shaping += 130 * pos.x / SCALE
    # 走到终点时的 x 约为 SCALE * TERRAIN_LENGTH ≈ 30 * 200 = 6000
    # shaping 最大约 130*6000/30 = 26000 → 最终累积奖励约 300

    # 惩罚头部倾斜（鼓励保持直立）
    shaping -= 5.0 * abs(state[0])  # state[0]=hull.angle

    # 步骤B：reward = shaping 的改善量（增量式）
    reward = shaping - prev_shaping
    # 每步向前移动 → 正奖励
    # 姿势变差 → 负奖励

    # 步骤C：惩罚关节力矩（鼓励节能行走）
    for a in action:
        reward -= 0.00035 * MOTORS_TORQUE * |a|
    # 四个关节的力矩越小 → 扣分越少 → 鼓励高效节能的步态

    # 步骤D：终止判断
    if game_over（摔倒）或 pos.x < 0（后退）:
        reward = -100; terminated = True

    if pos.x > 地形终点:
        terminated = True  # 成功走完全程
```

### 奖励函数数学本质

```
总奖励 = (本次_shaping - 上次_shaping)     ← 增量式：鼓励前进和维持直立
        - Σ 0.00035 × MOTORS_TORQUE × |a_i| ← 惩罚高力矩（鼓励节能）
        ← 摔倒或后退则 -100
```

**关键思想**：与 LunarLander 完全相同的 potential-based shaping 模式——奖励的是"改善量"，不是"绝对状态好"。

---

## BipedalWalker vs LunarLander 奖励对比

| | LunarLander | BipedalWalker |
|---|---|---|
| **动作空间** | Discrete(4) 离散 | Box(4) 连续 [-1,1] |
| **观测维度** | 8 维 | 24 维 |
| **shaping 项** | 距离+速度+角度+触地 | 前进距离+头部直立 |
| **节能惩罚** | m_power*0.3 + s_power*0.03 | Σ|a_i|*0.00035*TORQUE |
| **终止奖励** | 坠毁-100，着陆+100 | 摔倒-100，走完终点终止 |
| **奖励模式** | 增量式（shaping改善量） | 增量式（shaping改善量） |

---

## 三方法签名对比（BipedalWalker 版本）

| | 官方 Gymnasium | 我们 FDRE-HRDC |
|---|---|---|
| **签名** | 奖励内联在 step() 中 | `def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0): return float` |
| **obs/next_obs** | 单帧 24 维 | 双帧 24 维（可计算转移信号） |
| **action** | 4 维连续向量 | 同官方 |
| **training_progress** | 无 | 🆕 0→1，HRDC 阶段权重 |
