def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages movement toward target states)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large sudden changes, encourages smooth transitions)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalizes fuel usage)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 1.0
    elif action in [1, 3]:  # Side engines
        action_cost = 0.5
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on movement and exploration
    # Late training: focus on precision and efficiency
    stage1_weight = 1.0 - training_progress  # Decreases over time
    stage2_weight = training_progress  # Increases over time
    
    # Component rewards
    movement_reward = abs_change * stage1_weight
    smoothness_penalty = -squared_change * 0.1 * stage2_weight
    efficiency_penalty = -action_cost * 0.2 * (1.0 - training_progress * 0.5)
    
    # Ground contact bonus (from info or observation)
    ground_contact_bonus = 0.0
    if len(o) >= 8:
        # Last two elements are ground contact indicators
        if n[6] > 0.5 or n[7] > 0.5:
            ground_contact_bonus = 1.0 * stage2_weight
    
    # Combine components
    reward = movement_reward + smoothness_penalty + efficiency_penalty + ground_contact_bonus
    
    return reward