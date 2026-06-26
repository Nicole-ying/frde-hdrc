def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    
    # Feature 1: Change in absolute values (encourages moving towards center)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large sudden changes)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Feature 4: Progress-based weighting
    # Early training: focus on reducing absolute values (stabilizing)
    # Late training: focus on smooth transitions
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Combine features with stage weights
    reward = (
        early_weight * (0.5 * abs_diff - 0.1 * sq_change) +
        late_weight * (-0.2 * sq_change - action_cost)
    )
    
    return reward