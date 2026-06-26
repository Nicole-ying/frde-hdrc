def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude (change in absolute values)
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 2. Squared change (captures large deviations)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 3. Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on movement and exploration
    # Late stage: focus on stability and small changes
    w_movement = 1.0 - 0.8 * training_progress
    w_squared = 0.2 + 0.8 * training_progress
    w_action = 0.1 * (1.0 - training_progress)
    
    # Combine components
    reward = (
        w_movement * movement
        - w_squared * squared_change
        - w_action * action_cost
    )
    
    return float(reward)