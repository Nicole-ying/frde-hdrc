def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards center - encourage reducing absolute position values
    # Use sum of absolute differences between current and next state
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))
    movement_reward = pos_change * 0.5
    
    # Component 2: Smoothness - penalize large changes in state
    delta = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -delta * 0.1
    
    # Component 3: Action cost - small penalty for taking actions to encourage efficiency
    action_cost = -0.01 * float(action)
    
    # Component 4: Contact bonus - reward ground contact signals if present
    contact_bonus = 0.0
    if len(o) >= 8:
        left_contact = o[6]
        right_contact = o[7]
        contact_bonus = (left_contact + right_contact) * 0.5
    
    # Component 5: Velocity damping - reward reduction in velocity magnitude
    if len(o) >= 4:
        vel_curr = (o[2] ** 2 + o[3] ** 2) ** 0.5
        vel_next = (n[2] ** 2 + n[3] ** 2) ** 0.5
        velocity_damping = (vel_curr - vel_next) * 0.3
    else:
        velocity_damping = 0.0
    
    # Stage-based weighting
    if training_progress < 0.3:
        # Early stage: focus on exploration and movement
        w_movement = 1.0
        w_smoothness = 0.2
        w_action = 0.5
        w_contact = 0.1
        w_velocity = 0.3
    elif training_progress < 0.7:
        # Middle stage: balance all components
        w_movement = 0.8
        w_smoothness = 0.5
        w_action = 0.3
        w_contact = 0.5
        w_velocity = 0.6
    else:
        # Late stage: emphasize precision and stability
        w_movement = 0.5
        w_smoothness = 0.8
        w_action = 0.2
        w_contact = 0.8
        w_velocity = 0.8
    
    # Combine components
    reward = (w_movement * movement_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus + 
              w_velocity * velocity_damping)
    
    return reward