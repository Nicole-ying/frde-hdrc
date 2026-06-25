def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # --- Transition signals ---
    # 1. Directional improvement: reward moving toward zero (desirable state)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 2. Movement magnitude (squared differences) - for exploration
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 3. Action cost - penalize engine usage
    action_cost = 0.0
    if action == 2:  # main engine
        action_cost = 0.3
    elif action in [1, 3]:  # side engines
        action_cost = 0.15
    
    # 4. Survival bonus - reward staying alive
    survival_bonus = 0.05
    
    # 5. Stability signal - penalize large angular changes (using obs indices 4,5 for angle/angular velocity)
    # obs[4] is angle, obs[5] is angular velocity
    angle_change = abs(n[4] - o[4]) if len(o) > 4 else 0.0
    angvel_change = abs(n[5] - o[5]) if len(o) > 5 else 0.0
    stability_penalty = -(angle_change * 2.0 + angvel_change * 1.0)
    
    # 6. Height-based signal - reward moving toward ground (obs[1] is normalized height)
    # Positive when moving downward (toward landing)
    height_change = o[1] - n[1] if len(o) > 1 else 0.0
    
    # 7. Velocity penalty - penalize high horizontal velocity (obs[2] is horizontal velocity)
    horiz_vel = abs(n[2]) if len(o) > 2 else 0.0
    velocity_penalty = -horiz_vel * 0.5
    
    # --- Stage-based weights ---
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # --- Component rewards with proper scaling ---
    # Directional improvement: primary signal throughout
    abs_change_reward = abs_change * 0.8
    
    # Movement reward: encourage exploration early, discourage late
    movement_reward = movement * 0.2
    
    # Height descent reward: encourage moving down toward ground
    height_reward = height_change * 0.5
    
    # Action penalty
    action_penalty = -action_cost
    
    # --- Combine with stage weights ---
    # Early: explore with movement, reward directional improvement and height descent
    # Mid: focus on directional improvement and stability
    # Late: focus on stability and convergence
    reward = (
        early_weight * (abs_change_reward * 0.4 + movement_reward * 0.3 + height_reward * 0.3 + survival_bonus) +
        mid_weight * (abs_change_reward * 0.5 + height_reward * 0.3 + stability_penalty * 0.2 + survival_bonus) +
        late_weight * (abs_change_reward * 0.3 + height_reward * 0.2 + stability_penalty * 0.3 + velocity_penalty * 0.2 + survival_bonus) +
        action_penalty
    )
    
    return reward