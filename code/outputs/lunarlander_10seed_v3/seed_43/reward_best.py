def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # ========== SIGNAL CATEGORIES ==========

    # 1. TRANSITION PROGRESS: movement toward desirable state (directional)
    # Use sum of absolute values decreasing as progress signal
    progress = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # 2. SMOOTHNESS: penalize large sudden changes (L2 norm of delta)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # 3. ACTION COST: penalize engine usage
    # action 0 = no engine, 1/3 = side engine, 2 = main engine
    action_cost = 0.0
    if action == 2:
        action_cost = -0.3  # main engine is expensive
    elif action == 1 or action == 3:
        action_cost = -0.1  # side engine is moderate
    # action 0 costs nothing

    # 4. STABILITY: penalize high angular velocity and angle
    # obs[4] = angle, obs[5] = angular velocity (scaled)
    angle_penalty = -abs(n[4])  # penalize being tilted
    angvel_penalty = -abs(n[5])  # penalize spinning

    # 5. VELOCITY PENALTY: penalize high speed (especially vertical)
    # obs[2] = x velocity, obs[3] = y velocity
    speed_penalty = -(n[2]**2 + n[3]**2) * 0.2

    # 6. CONTACT BONUS: reward ground contact (legs)
    # obs[6] and obs[7] are leg contact flags
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = n[6] + n[7]  # 0, 1, or 2 legs in contact

    # 7. SURVIVAL BONUS: reward staying alive
    survival_bonus = 0.05

    # ========== STAGE WEIGHTS ==========
    # Early: explore, learn to move and stabilize
    # Middle: balance all signals
    # Late: prioritize precision landing

    if training_progress < 0.3:
        # Early stage: focus on stability and gentle movement
        w_progress = 0.3
        w_smoothness = 0.1
        w_action = 0.05
        w_angle = 0.3
        w_angvel = 0.3
        w_speed = 0.1
        w_contact = 0.2
        w_survival = 0.3
    elif training_progress < 0.7:
        # Middle stage: balanced
        w_progress = 0.4
        w_smoothness = 0.1
        w_action = 0.05
        w_angle = 0.2
        w_angvel = 0.2
        w_speed = 0.1
        w_contact = 0.3
        w_survival = 0.2
    else:
        # Late stage: precision landing
        w_progress = 0.5
        w_smoothness = 0.05
        w_action = 0.05
        w_angle = 0.15
        w_angvel = 0.15
        w_speed = 0.05
        w_contact = 0.4
        w_survival = 0.1

    # ========== COMPUTE REWARD ==========
    reward = 0.0
    reward += w_progress * progress
    reward += w_smoothness * smoothness
    reward += w_action * action_cost
    reward += w_angle * angle_penalty
    reward += w_angvel * angvel_penalty
    reward += w_speed * speed_penalty
    reward += w_contact * contact_bonus
    reward += w_survival * survival_bonus

    # Scale to keep values in reasonable range
    scale = 1.0 / (1.0 + len(o) * 0.05)
    reward = reward * scale

    return reward