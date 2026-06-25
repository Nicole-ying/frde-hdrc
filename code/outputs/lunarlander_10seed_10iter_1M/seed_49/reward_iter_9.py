def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large changes in absolute values
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_change * 0.1
    
    # Component 2: Smoothness - penalize large squared differences (encourages gentle transitions)
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.05
    
    # Component 3: Action cost - penalize taking actions (encourages efficiency)
    action_cost = -0.01 * float(action)
    
    # Component 4: Contact bonus - reward ground contact signals (indices 6 and 7 in observation)
    contact_bonus = 0.0
    if len(o) >= 8:
        prev_contact = o[6] + o[7]
        curr_contact = n[6] + n[7]
        contact_bonus = (curr_contact - prev_contact) * 0.5
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on movement and exploration
    # Mid training: balance movement and smoothness
    # Late training: emphasize contact and efficiency
    
    if training_progress < 0.3:
        # Early stage: encourage movement and exploration
        w_movement = 1.0
        w_smoothness = 0.3
        w_action = 0.5
        w_contact = 0.2
    elif training_progress < 0.7:
        # Middle stage: balance all components
        w_movement = 0.7
        w_smoothness = 0.7
        w_action = 0.3
        w_contact = 0.6
    else:
        # Late stage: focus on stability and contact
        w_movement = 0.3
        w_smoothness = 1.0
        w_action = 0.2
        w_contact = 1.0
    
    # Combine components with stage weights
    reward = (w_movement * movement_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    return reward