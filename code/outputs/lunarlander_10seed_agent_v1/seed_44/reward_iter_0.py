def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Create generic arrays for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large absolute changes
    # Encourages smooth transitions and stable states
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff / len(o)
    
    # Component 2: Smoothness - penalize sudden large changes
    # Uses squared differences to heavily penalize erratic movements
    sq_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -sq_diff / (len(o) * 10.0)
    
    # Component 3: Action cost - penalize frequent or aggressive actions
    # Actions are discrete (0-3), normalize to [-1, 1] range
    action_value = (action - 1.5) / 1.5
    action_cost = -abs(action_value) * 0.1
    
    # Component 4: Contact bonus - reward ground contact signals from info
    # Generic contact detection from observation indices 6 and 7
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = (n[6] + n[7]) * 0.5  # Reward both legs contacting
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on smoothness and movement
    # Middle stage: balance all components
    # Late stage: emphasize stability and contact
    
    if training_progress < 0.3:
        # Early exploration phase
        w_movement = 1.0
        w_smoothness = 2.0
        w_action = 0.5
        w_contact = 0.0
    elif training_progress < 0.6:
        # Middle refinement phase
        w_movement = 0.5
        w_smoothness = 1.0
        w_action = 1.0
        w_contact = 1.0
    else:
        # Late optimization phase
        w_movement = 0.2
        w_smoothness = 0.5
        w_action = 1.5
        w_contact = 2.0
    
    # Combine components with stage weights
    reward = (w_movement * movement_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    return reward