def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages moving towards zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large jumps)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Stage-based weights
    # Early stage: focus on reducing absolute values
    # Middle stage: balance between reduction and stability
    # Late stage: focus on stability (small changes)
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage weights
    reward = (
        early_weight * abs_change * 0.5 +
        mid_weight * (-squared_change * 0.1) +
        late_weight * (-squared_change * 0.2) -
        action_cost * (0.5 + 0.5 * training_progress)
    )
    
    return reward