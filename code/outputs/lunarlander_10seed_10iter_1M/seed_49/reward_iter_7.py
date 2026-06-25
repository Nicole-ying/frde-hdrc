def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability - penalize large changes in absolute values
    # This encourages smooth transitions
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.1
    
    # Component 2: Smoothness - penalize large squared changes
    # This encourages consistent behavior
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.05
    
    # Component 3: Action cost - penalize taking actions
    # This encourages efficiency
    action_cost = -0.01 * float(action)
    
    # Component 4: Ground contact bonus from info
    # Generic signal that might indicate progress
    contact_bonus = 0.0
    if info and 'ground_contact' in info:
        contact_bonus = 0.5 * float(info['ground_contact'])
    
    # Stage weights based on training_progress
    # Early training: focus on movement and exploration
    # Late training: focus on smoothness and efficiency
    stage_weight_movement = 1.0 - 0.5 * training_progress
    stage_weight_smoothness = 0.5 + 0.5 * training_progress
    stage_weight_action = 0.3 * (1.0 - training_progress)
    stage_weight_contact = 0.8 * training_progress
    
    # Combine components with stage weights
    reward = (
        stage_weight_movement * movement_reward +
        stage_weight_smoothness * smoothness_penalty +
        stage_weight_action * action_cost +
        stage_weight_contact * contact_bonus
    )
    
    return reward