def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Feature 1: Movement toward zero (directional - rewards reducing absolute values)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness penalty (penalizes large state changes)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalizes large actions)
    action_cost = float(action) ** 2
    
    # Feature 4: Survival bonus (rewards staying alive)
    survival_bonus = 1.0
    
    # Stage weights based on training progress
    w1 = 1.0 - 0.3 * training_progress  # abs_change weight
    w2 = 0.1 + 0.3 * training_progress   # squared_change weight
    w3 = 0.01 * (1.0 - training_progress)  # action_cost weight
    w4 = 0.5 + 0.5 * training_progress  # survival bonus weight
    
    reward = w1 * abs_change - w2 * squared_change - w3 * action_cost + w4 * survival_bonus
    
    return reward