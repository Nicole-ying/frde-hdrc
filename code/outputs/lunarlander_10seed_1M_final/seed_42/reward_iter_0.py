def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability (reducing absolute values)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_change * 0.1
    
    # Component 2: Smoothness of transition (small changes are better)
    diff_sq = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_reward = -diff_sq * 0.5
    
    # Component 3: Action penalty (encourage efficient actions)
    action_cost = -0.01 * float(action)
    
    # Component 4: Contact bonus (from info if available, otherwise from obs)
    contact_bonus = 0.0
    if len(o) >= 8:
        leg1_contact = o[6]
        leg2_contact = o[7]
        next_leg1 = n[6]
        next_leg2 = n[7]
        # Reward gaining ground contact
        contact_bonus = (next_leg1 - leg1_contact) * 0.5 + (next_leg2 - leg2_contact) * 0.5
    
    # Stage weights based on training_progress
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)
    stage2_weight = 1.0 - abs(training_progress - 0.5) * 2.0
    stage3_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage-adaptive weights
    reward = (
        stage1_weight * movement_reward +
        stage2_weight * smoothness_reward +
        stage3_weight * contact_bonus +
        action_cost * (1.0 - training_progress * 0.5)
    )
    
    return reward