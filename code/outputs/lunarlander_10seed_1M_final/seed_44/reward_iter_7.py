def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large changes in absolute values
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness - penalize large squared differences
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Component 3: Action cost - penalize taking actions (encourage efficiency)
    action_cost = 0.0
    if action == 2 or action == 1 or action == 3:
        action_cost = -0.1
    
    # Stage-based weights that evolve with training
    # Early training: focus on exploration (encourage movement)
    # Mid training: balance stability and action efficiency
    # Late training: focus on precision and stability
    
    if training_progress < 0.3:
        # Early stage: encourage exploration and movement
        w1 = 0.3  # abs_change weight
        w2 = -0.1  # penalize large changes slightly
        w3 = -0.05  # small action penalty
    elif training_progress < 0.7:
        # Mid stage: balance exploration with stability
        w1 = 0.5
        w2 = -0.3
        w3 = -0.1
    else:
        # Late stage: focus on stability and precision
        w1 = 0.7
        w2 = -0.5
        w3 = -0.15
    
    # Combine components
    reward = w1 * abs_change + w2 * squared_diff + w3 * action_cost
    
    # Add small constant to encourage survival
    reward += 0.01
    
    return reward