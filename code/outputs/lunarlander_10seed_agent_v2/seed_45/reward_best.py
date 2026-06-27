def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large jumps)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalizes action usage)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0  # no action
    elif action == 1 or action == 3:
        action_cost = 0.1  # side engine
    else:  # action == 2
        action_cost = 0.2  # main engine
    
    # Feature 4: Ground contact bonus
    ground_contact = 0.0
    if len(o) >= 8:
        ground_contact = n[6] + n[7]  # sum of leg contacts
    
    # Stage-based weights that evolve with training progress
    stage1_weight = 1.0 - training_progress  # early: focus on movement
    stage2_weight = training_progress  # later: focus on stability
    
    # Combine components with stage weights
    reward = (
        stage1_weight * (abs_change * 0.5 - squared_change * 0.1 - action_cost * 0.3)
        + stage2_weight * (ground_contact * 2.0 - squared_change * 0.05)
    )
    
    return reward