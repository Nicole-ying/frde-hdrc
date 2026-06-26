def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract arrays for generic processing
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement towards center (sum of absolute values reduction)
    abs_diff = sum(abs(o[i]) for i in range(len(o))) - sum(abs(n[i]) for i in range(len(n)))
    
    # Feature 2: Smoothness (small changes are better)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (encourage efficient control)
    action_cost = -0.01 * action
    
    # Stage weights based on training progress
    # Early exploration: focus on movement and smoothness
    # Later exploitation: maintain smoothness with less action cost
    w1 = 1.0 - 0.5 * training_progress  # weight for abs_diff
    w2 = 0.5 + 0.5 * training_progress  # weight for smoothness
    w3 = 0.1 * (1.0 - training_progress)  # weight for action cost
    
    # Combine components
    reward = w1 * abs_diff + w2 * smoothness + w3 * action_cost
    
    return reward