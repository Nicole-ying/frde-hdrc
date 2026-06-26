def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract arrays for generic processing
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability - reward reduction in absolute values
    # This encourages the agent to bring state variables closer to zero
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(min(len(o), len(n))))
    movement_reward = abs_diff * 0.1
    
    # Component 2: Smoothness - penalize large changes in state
    # This encourages smooth, controlled movements
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(min(len(o), len(n))))
    smoothness_penalty = -squared_diff * 0.05
    
    # Component 3: Action cost - penalize taking actions to encourage efficiency
    action_cost = -0.01 * float(action)
    
    # Component 4: Progress bonus - reward based on info if available
    progress_bonus = 0.0
    if info and isinstance(info, dict):
        if 'progress' in info:
            progress_bonus = float(info['progress']) * 0.2
    
    # Stage-based weighting
    # Early training: focus on exploration and movement
    # Mid training: balance movement and smoothness
    # Late training: focus on precision and efficiency
    if training_progress < 0.3:
        # Early stage - encourage exploration and movement
        w1 = 1.0  # movement reward
        w2 = 0.3  # smoothness penalty
        w3 = 0.1  # action cost
        w4 = 0.0  # progress bonus
    elif training_progress < 0.7:
        # Middle stage - balance all components
        w1 = 0.7
        w2 = 0.6
        w3 = 0.3
        w4 = 0.5
    else:
        # Late stage - focus on precision and efficiency
        w1 = 0.3
        w2 = 1.0
        w3 = 0.5
        w4 = 1.0
    
    # Combine components with stage weights
    reward = (w1 * movement_reward + 
              w2 * smoothness_penalty + 
              w3 * action_cost + 
              w4 * progress_bonus)
    
    return reward