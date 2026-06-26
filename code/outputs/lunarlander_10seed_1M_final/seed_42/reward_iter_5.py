def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large absolute changes
    # Use sum of absolute differences between current and next observation
    abs_diff = sum(abs(o[i] - n[i]) for i in range(len(o)))
    movement_penalty = -abs_diff * 0.01
    
    # Component 2: Convergence signal - reward when observations move toward zero
    # Sum of (abs(o) - abs(n)) indicates moving toward or away from origin
    convergence = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    convergence_reward = convergence * 0.1
    
    # Component 3: Smoothness - penalize large squared changes (jerkiness)
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.005
    
    # Component 4: Action penalty - discourage excessive action usage
    action_cost = -0.02 * float(action)
    
    # Component 5: Contact bonus - reward ground contact signals (indices 6,7 in obs)
    contact_bonus = 0.0
    if len(o) >= 8:
        current_contact = o[6] + o[7]
        next_contact = n[6] + n[7]
        contact_bonus = (next_contact - current_contact) * 0.5
    
    # Stage weights based on training_progress (0.0 to 1.0)
    # Early stage: focus on movement and convergence
    # Middle stage: balance all components
    # Late stage: emphasize smoothness and contact
    
    if training_progress < 0.3:
        # Early exploration
        w_movement = 1.0
        w_convergence = 0.5
        w_smoothness = 0.2
        w_action = 0.3
        w_contact = 0.1
    elif training_progress < 0.7:
        # Middle refinement
        w_movement = 0.5
        w_convergence = 1.0
        w_smoothness = 0.5
        w_action = 0.5
        w_contact = 0.5
    else:
        # Late optimization
        w_movement = 0.2
        w_convergence = 0.5
        w_smoothness = 1.0
        w_action = 0.8
        w_contact = 1.0
    
    # Combine components with stage weights
    reward = (
        w_movement * movement_penalty +
        w_convergence * convergence_reward +
        w_smoothness * smoothness_penalty +
        w_action * action_cost +
        w_contact * contact_bonus
    )
    
    # Add small constant to encourage exploration
    reward += 0.01
    
    return reward