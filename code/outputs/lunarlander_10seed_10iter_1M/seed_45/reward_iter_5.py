def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - reward reduction in absolute values
    # This encourages the agent to move toward zero (stable state) without knowing what zero means
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_change / len(o)
    
    # Component 2: Smoothness penalty - penalize large changes between steps
    # This encourages smooth, controlled actions
    diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -diff / (len(o) * 10.0)
    
    # Component 3: Action cost - penalize engine usage to encourage efficiency
    # Action is discrete (0,1,2,3), penalize non-zero actions
    action_cost = -0.1 if action != 0 else 0.0
    
    # Component 4: Contact bonus - reward ground contact (from info if available)
    # Use info to detect if legs are making contact (indices 6 and 7 in obs)
    contact_bonus = 0.0
    if len(o) >= 8:
        leg1_contact = o[6]
        leg2_contact = o[7]
        contact_bonus = 0.5 * (leg1_contact + leg2_contact)
    
    # Stage weights based on training_progress
    # Early stage: focus on movement and smoothness
    # Middle stage: balance all components
    # Late stage: emphasize contact and efficiency
    if training_progress < 0.3:
        # Early exploration
        w_movement = 1.0
        w_smoothness = 0.5
        w_action = 0.2
        w_contact = 0.1
    elif training_progress < 0.7:
        # Middle refinement
        w_movement = 0.8
        w_smoothness = 0.8
        w_action = 0.5
        w_contact = 0.5
    else:
        # Late optimization
        w_movement = 0.5
        w_smoothness = 0.3
        w_action = 0.8
        w_contact = 1.0
    
    # Combine components
    reward = (w_movement * movement_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    return reward