def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Dimension-agnostic transition features
    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large sudden changes)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (small penalty for taking actions)
    action_cost = 0.01 * action
    
    # Stage weights based on training progress
    # Early exploration: encourage movement and exploration
    # Late exploitation: encourage stability and precision
    w1 = 1.0 - 0.5 * training_progress  # weight for abs_change
    w2 = 0.1 + 0.9 * training_progress  # weight for stability
    w3 = 0.1  # constant action penalty
    
    # Combine components
    reward = w1 * abs_change - w2 * sq_change - w3 * action_cost
    
    return reward