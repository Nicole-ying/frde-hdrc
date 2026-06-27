def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement toward lower absolute values (stabilization)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness - penalize large changes (encourages gentle control)
    delta_sq = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost - penalize engine usage
    action_cost = 0.0
    if action == 1 or action == 3:
        action_cost = 0.1
    elif action == 2:
        action_cost = 0.2
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on stabilization and smoothness
    # Late stage: focus on precision and efficiency
    stage1_weight = 1.0 - training_progress  # Early stage dominance
    stage2_weight = training_progress        # Late stage dominance
    
    # Component weights
    w_abs = 0.5 + 0.5 * stage1_weight  # Stabilization weight decreases over time
    w_smooth = 0.3 + 0.3 * stage2_weight  # Smoothness weight increases over time
    w_action = 0.2 + 0.2 * stage2_weight  # Action penalty increases over time
    
    # Compute reward components
    reward_abs = w_abs * abs_diff
    reward_smooth = -w_smooth * delta_sq * 0.1  # Scale down to avoid large negative values
    reward_action = -w_action * action_cost
    
    # Combine components
    reward = reward_abs + reward_smooth + reward_action
    
    # Add small positive bias for stability
    reward = reward + 0.01
    
    return reward