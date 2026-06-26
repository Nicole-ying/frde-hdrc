def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large jumps, rewards smooth transitions)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small action cost to encourage efficiency)
    action_cost = 0.01 * float(action)
    
    # Stage weights based on training progress
    # Early stage: focus on reducing absolute values (stabilization)
    # Middle stage: balance between stabilization and smoothness
    # Late stage: fine-tuning with small action cost
    w1 = 1.0 - 0.5 * training_progress  # weight for abs_change
    w2 = 0.1 + 0.4 * training_progress   # weight for squared_change
    w3 = 0.01 * (1.0 - training_progress)  # weight for action_cost
    
    # Combine components
    reward = w1 * abs_change - w2 * squared_change - w3 * action_cost
    
    return reward