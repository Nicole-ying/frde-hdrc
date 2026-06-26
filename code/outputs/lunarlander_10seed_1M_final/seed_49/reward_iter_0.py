def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement magnitude (change in absolute values)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (smoothness/stability)
    sq_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small action bias)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0  # no action
    elif action == 2:
        action_cost = 0.1  # main engine
    else:
        action_cost = 0.05  # side engines
    
    # Stage-based weights
    # Early stage: focus on movement and exploration
    # Late stage: focus on stability and precision
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Component weights that evolve with training
    w_abs = 0.5 * early_weight + 0.1 * late_weight
    w_sq = 0.1 * early_weight + 0.5 * late_weight
    w_action = 0.4 * early_weight + 0.4 * late_weight
    
    # Compute reward components
    reward_abs = abs_diff * w_abs
    reward_sq = -sq_diff * w_sq  # penalize large changes (stability)
    reward_action = -action_cost * w_action
    
    # Combine components
    reward = reward_abs + reward_sq + reward_action
    
    return float(reward)