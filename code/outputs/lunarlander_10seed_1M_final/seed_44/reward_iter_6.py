def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages reaching stable states)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large transitions, encourages smoothness)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalizes engine usage)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 1.0
    elif action in [1, 3]:  # Side engines
        action_cost = 0.5
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on reducing absolute values and smooth transitions
    w_abs = 1.0 - 0.5 * training_progress
    # Mid training: gradually reduce smoothness penalty
    w_sq = 0.5 * (1.0 - training_progress)
    # Late training: reduce action cost to allow fine-tuning
    w_action = 0.2 * (1.0 - 0.8 * training_progress)
    
    # Combine components
    reward = w_abs * abs_change - w_sq * sq_change - w_action * action_cost
    
    return reward