def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Dimension-agnostic transition features
    # Movement toward target: sum of absolute differences
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Smoothness: squared change
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Stage weights based on training progress
    # Early: focus on movement toward target
    # Late: focus on stability (small changes)
    w1 = 1.0 - 0.5 * training_progress  # weight for abs_diff
    w2 = 0.5 * training_progress        # weight for squared_change
    w3 = 0.1                            # constant action cost weight
    
    # Combine components
    reward = w1 * abs_diff - w2 * squared_change - w3 * action_cost
    
    return float(reward)