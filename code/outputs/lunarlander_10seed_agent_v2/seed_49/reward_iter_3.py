def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement towards stability - sum of absolute differences
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness of transition - squared differences
    sq_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Stage-based weights
    # Early stage: focus on exploration and reducing absolute values
    w1 = 0.5 * (1.0 - training_progress)
    # Middle stage: focus on smooth transitions
    w2 = 0.3 * (1.0 - abs(training_progress - 0.5) * 2)
    # Late stage: focus on precision
    w3 = 0.2 * training_progress
    
    # Combine components
    reward = w1 * abs_diff - w2 * sq_diff - w3 * action_cost
    
    return float(reward)