def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement toward zero (reducing absolute values)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness of transition (small changes are better)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalize large actions)
    action_cost = -float(action) * 0.01
    
    # Stage weights based on training progress
    # Early stage: focus on reducing absolute values (getting to stable region)
    # Middle stage: focus on smooth transitions
    # Late stage: balance both with more smoothness emphasis
    if training_progress < 0.3:
        w1 = 1.0
        w2 = 0.1
        w3 = 0.05
    elif training_progress < 0.7:
        w1 = 0.6
        w2 = 0.4
        w3 = 0.1
    else:
        w1 = 0.3
        w2 = 0.7
        w3 = 0.15
    
    # Combine components
    reward = w1 * abs_diff + w2 * smoothness + w3 * action_cost
    
    return reward