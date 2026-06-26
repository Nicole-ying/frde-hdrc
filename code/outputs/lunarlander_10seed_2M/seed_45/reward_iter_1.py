def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward zero (stability) - directional
    # Reward reduction in absolute values across all state dimensions
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.2
    
    # Component 2: Velocity penalty - penalize high speeds (indices 2,3 are velocity)
    # This prevents crashing at high speed and encourages gentle landing
    vel_penalty = 0.0
    if len(o) >= 4:
        prev_vel = (o[2] ** 2 + o[3] ** 2) ** 0.5
        curr_vel = (n[2] ** 2 + n[3] ** 2) ** 0.5
        vel_penalty = -curr_vel * 0.1  # Penalize current speed
    
    # Component 3: Angle penalty - penalize tilting (index 4 is angle)
    # This prevents tumbling and encourages upright orientation
    angle_penalty = 0.0
    if len(o) >= 5:
        angle_penalty = -abs(n[4]) * 0.3  # Penalize absolute angle
    
    # Component 4: Contact bonus - reward sustained ground contact
    # Last two dimensions are contact indicators (0 or 1)
    contact_bonus = 0.0
    if len(o) >= 8:
        curr_contact = n[-2] + n[-1]
        # Reward having contact (both legs on ground)
        contact_bonus = curr_contact * 0.5
    
    # Component 5: Smoothness penalty - penalize large state changes
    # This encourages smooth transitions
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.02
    
    # Component 6: Action cost - small penalty for using engines
    # Convert discrete action to a small cost
    action_cost = -0.02 * (1.0 if action != 0 else 0.0)
    
    # Stage-based weighting
    # Early training: focus on movement and exploration
    # Mid training: balance all components
    # Late training: emphasize precision and landing
    if training_progress < 0.3:
        # Early stage - encourage exploration and movement
        w_movement = 1.0
        w_vel = 0.3
        w_angle = 0.5
        w_contact = 0.2
        w_smoothness = 0.3
        w_action = 0.5
    elif training_progress < 0.7:
        # Mid stage - balance components
        w_movement = 0.7
        w_vel = 0.7
        w_angle = 0.8
        w_contact = 0.6
        w_smoothness = 0.5
        w_action = 0.3
    else:
        # Late stage - emphasize precision, stability, and contact
        w_movement = 0.3
        w_vel = 1.0
        w_angle = 1.0
        w_contact = 1.0
        w_smoothness = 0.7
        w_action = 0.1
    
    # Combine components
    reward = (w_movement * movement_reward + 
              w_vel * vel_penalty + 
              w_angle * angle_penalty + 
              w_contact * contact_bonus + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost)
    
    return reward