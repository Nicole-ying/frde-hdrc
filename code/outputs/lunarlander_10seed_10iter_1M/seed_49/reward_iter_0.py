def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement towards center - sum of absolute position changes
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Feature 2: Smoothness - squared difference in velocities
    vel_diff = sum((n[i] - o[i]) ** 2 for i in range(2, 4))
    
    # Feature 3: Angular stability - change in angle and angular velocity
    angle_change = abs(n[4]) - abs(o[4])
    ang_vel_change = abs(n[5]) - abs(o[5])
    
    # Feature 4: Ground contact bonus
    ground_contact = sum(n[6:8]) - sum(o[6:8])
    
    # Feature 5: Action cost (small penalty for taking actions)
    action_cost = 0.01 * abs(action - 1.5)  # Center actions around 1.5
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on movement and exploration
    w_pos = 1.0 + 2.0 * training_progress  # Increase importance of position over time
    w_vel = 0.5 * (1.0 - training_progress)  # Decrease velocity smoothness importance
    w_angle = 0.3 * (1.0 - 0.5 * training_progress)  # Slowly decrease angle importance
    w_contact = 2.0 * training_progress  # Increase ground contact importance
    w_action = 0.1 * (1.0 - 0.5 * training_progress)  # Decrease action cost over time
    
    # Combine components
    reward = (
        w_pos * pos_change
        - w_vel * vel_diff
        + w_angle * (angle_change + ang_vel_change)
        + w_contact * ground_contact
        - w_action * action_cost
    )
    
    return reward