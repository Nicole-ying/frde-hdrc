def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement magnitude - change in absolute values
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (captures large deviations)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalize frequent actions)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0
    elif action == 1 or action == 3:
        action_cost = 0.1
    elif action == 2:
        action_cost = 0.2
    
    # Feature 4: Stability signal - check if legs are in contact (indices 6 and 7)
    leg_contact = 0.0
    if len(n) >= 8:
        leg_contact = n[6] + n[7]  # Sum of leg contact indicators
    
    # Feature 5: Velocity change (indices 2 and 3 are velocities)
    vel_change = 0.0
    if len(n) >= 4 and len(o) >= 4:
        vel_change = abs(n[2] - o[2]) + abs(n[3] - o[3])
    
    # Stage weights based on training_progress
    # Early stage: focus on exploration and movement
    # Middle stage: balance movement with stability
    # Late stage: emphasize stability and precision
    
    if training_progress < 0.3:
        # Early training: encourage exploration and movement
        w_movement = 1.0
        w_squared = 0.5
        w_action = -0.2
        w_contact = 0.3
        w_vel = 0.1
    elif training_progress < 0.7:
        # Middle training: balance exploration with stability
        w_movement = 0.6
        w_squared = 0.8
        w_action = -0.3
        w_contact = 0.6
        w_vel = 0.2
    else:
        # Late training: emphasize stability and precision
        w_movement = 0.3
        w_squared = 1.0
        w_action = -0.4
        w_contact = 1.0
        w_vel = 0.3
    
    # Combine components
    reward = (
        w_movement * movement +
        w_squared * squared_change +
        w_action * action_cost +
        w_contact * leg_contact +
        w_vel * vel_change
    )
    
    return reward