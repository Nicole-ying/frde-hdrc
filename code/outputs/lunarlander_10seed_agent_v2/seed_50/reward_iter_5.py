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
    ground_contact = sum(n[6:8])
    
    # Feature 5: Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance stability and control
    # Late stage: emphasize precision and ground contact
    
    if training_progress < 0.3:
        # Early training: encourage exploration and movement
        w_pos = 1.0
        w_vel = -0.5
        w_angle = 0.3
        w_ang_vel = -0.2
        w_contact = 0.5
        w_action = -0.1
    elif training_progress < 0.7:
        # Middle training: balance stability and control
        w_pos = 0.7
        w_vel = -0.8
        w_angle = 0.5
        w_ang_vel = -0.4
        w_contact = 1.0
        w_action = -0.05
    else:
        # Late training: precision and landing
        w_pos = 0.3
        w_vel = -1.0
        w_angle = 0.7
        w_ang_vel = -0.6
        w_contact = 2.0
        w_action = -0.02
    
    # Compute reward components
    reward = 0.0
    reward += w_pos * pos_change
    reward += w_vel * vel_diff
    reward += w_angle * angle_change
    reward += w_ang_vel * ang_vel_change
    reward += w_contact * ground_contact
    reward += w_action * action_cost
    
    return reward