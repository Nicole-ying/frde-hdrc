def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs

    # ========== TRANSITION SIGNALS ==========

    # 1. Movement toward center (directional: decreasing absolute values is good)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # 2. Smoothness penalty (discourage large jerky movements)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # 3. Action penalty (discourage unnecessary engine use)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3

    # 4. Contact bonus (reward ground contact)
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = (o[6] + o[7]) * 0.5

    # 5. Stability signal (penalize angular velocity and angle)
    angle_penalty = 0.0
    angvel_penalty = 0.0
    if len(o) >= 6:
        angle_penalty = -abs(o[4])  # penalize being tilted
        angvel_penalty = -abs(o[5])  # penalize spinning

    # 6. Velocity penalty (discourage high speed crashes)
    vel_penalty = 0.0
    if len(o) >= 4:
        vel_penalty = -(abs(o[2]) + abs(o[3]))

    # ========== STAGE WEIGHTS ==========
    stage1_end = 0.3
    stage2_end = 0.7

    if training_progress < stage1_end:
        # Early stage: explore, learn to move and stabilize
        w_abs_change = 0.5
        w_smoothness = 0.05
        w_action = -0.05
        w_contact = 0.1
        w_angle = 0.15
        w_angvel = 0.15
        w_vel = 0.05
    elif training_progress < stage2_end:
        # Middle stage: balance exploration with control
        w_abs_change = 0.4
        w_smoothness = 0.15
        w_action = -0.2
        w_contact = 0.25
        w_angle = 0.25
        w_angvel = 0.25
        w_vel = 0.15
    else:
        # Late stage: precision landing, stability critical
        w_abs_change = 0.3
        w_smoothness = 0.25
        w_action = -0.4
        w_contact = 0.6
        w_angle = 0.35
        w_angvel = 0.35
        w_vel = 0.25

    # ========== COMPUTE REWARD ==========
    reward = (w_abs_change * abs_change +
              w_smoothness * smoothness +
              w_action * action_cost +
              w_contact * contact_bonus +
              w_angle * angle_penalty +
              w_angvel * angvel_penalty +
              w_vel * vel_penalty)

    # Small exploration bonus that decays
    exploration = 0.01 * sum(abs(n[i] - o[i]) for i in range(len(o)))
    reward += exploration * max(0.0, 1.0 - 2.0 * training_progress)

    return float(reward)