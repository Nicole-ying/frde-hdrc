def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large changes in absolute values
    # This encourages the agent to reduce extreme state values
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.1
    
    # Component 2: Smoothness penalty - penalize large squared changes
    # This encourages smooth transitions and discourages erratic behavior
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.05
    
    # Component 3: Action cost - penalize taking actions to encourage efficiency
    action_cost = -0.01 * float(action)
    
    # Component 4: Contact bonus - reward ground contact (leg sensors are at indices 6 and 7)
    contact_bonus = 0.0
    if len(o) >= 8:
        leg1_contact = n[6]
        leg2_contact = n[7]
        contact_bonus = (leg1_contact + leg2_contact) * 0.5
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on movement and smoothness
    # Mid training: balance all components
    # Late training: emphasize contact and efficiency
    
    if training_progress < 0.3:
        # Early stage - learn basic control
        w_movement = 1.0
        w_smoothness = 0.5
        w_action = 0.1
        w_contact = 0.2
    elif training_progress < 0.6:
        # Middle stage - refine control
        w_movement = 0.7
        w_smoothness = 0.8
        w_action = 0.3
        w_contact = 0.5
    else:
        # Late stage - optimize for landing
        w_movement = 0.3
        w_smoothness = 0.5
        w_action = 0.2
        w_contact = 1.0
    
    # Combine components with stage weights
    reward = (w_movement * movement_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    return reward