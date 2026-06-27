def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages moving towards zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large changes, encourages smooth transitions)
    squared_change = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small action cost to encourage efficiency)
    action_cost = -0.01 * float(action)
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on reducing absolute values
    # Mid training: balance between absolute reduction and smoothness
    # Late training: fine-tune with action efficiency
    if training_progress < 0.3:
        w_abs = 1.0
        w_sq = 0.1
        w_action = 0.01
    elif training_progress < 0.7:
        w_abs = 0.6
        w_sq = 0.4
        w_action = 0.1
    else:
        w_abs = 0.3
        w_sq = 0.5
        w_action = 0.2
    
    # Combine components
    reward = w_abs * abs_change + w_sq * squared_change + w_action * action_cost
    
    return float(reward)