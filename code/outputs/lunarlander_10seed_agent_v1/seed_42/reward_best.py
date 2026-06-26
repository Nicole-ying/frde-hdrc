def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components as generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages moving towards zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large jumps, encourages smoothness)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (discourage excessive action usage)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0  # no action
    elif action in [1, 3]:
        action_cost = 0.1  # side engines
    else:  # action == 2
        action_cost = 0.2  # main engine
    
    # Stage-based weights that evolve with training
    # Early stage: focus on reducing absolute values and smooth transitions
    # Late stage: focus on fine control and minimizing action cost
    stage1_weight = 1.0 - training_progress  # early stage weight
    stage2_weight = training_progress  # late stage weight
    
    # Component weights
    w_abs = 0.5 * stage1_weight + 0.1 * stage2_weight
    w_sq = -0.1 * stage1_weight - 0.3 * stage2_weight
    w_action = -0.2 * stage1_weight - 0.5 * stage2_weight
    
    # Combine components
    reward = w_abs * abs_change + w_sq * sq_change + w_action * action_cost
    
    # Add small exploration bonus in early stages
    exploration_bonus = 0.01 * (1.0 - training_progress)
    reward += exploration_bonus
    
    return reward