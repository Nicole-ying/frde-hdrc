def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Extract position and velocity components
    pos_x_o, pos_y_o = o[0], o[1]
    vel_x_o, vel_y_o = o[2], o[3]
    angle_o = o[4]
    ang_vel_o = o[5]
    leg1_o, leg2_o = o[6], o[7]
    
    pos_x_n, pos_y_n = n[0], n[1]
    vel_x_n, vel_y_n = n[2], n[3]
    angle_n = n[4]
    ang_vel_n = n[5]
    leg1_n, leg2_n = n[6], n[7]
    
    # Reward for moving towards center (x position)
    center_approach = abs(pos_x_o) - abs(pos_x_n)
    
    # Reward for reducing velocity magnitude
    vel_mag_o = (vel_x_o ** 2 + vel_y_o ** 2) ** 0.5
    vel_mag_n = (vel_x_n ** 2 + vel_y_n ** 2) ** 0.5
    velocity_reduction = vel_mag_o - vel_mag_n
    
    # Reward for reducing angle magnitude
    angle_reduction = abs(angle_o) - abs(angle_n)
    
    # Reward for reducing angular velocity
    ang_vel_reduction = abs(ang_vel_o) - abs(ang_vel_n)
    
    # Reward for ground contact (landing legs)
    ground_contact = leg1_n + leg2_n
    
    # Penalty for action (fuel usage)
    action_penalty = 0.0
    if action == 2:  # Main engine
        action_penalty = 0.1
    elif action == 1 or action == 3:  # Side engines
        action_penalty = 0.05
    
    # Reward for being close to ground (y position near 0)
    height_reward = -abs(pos_y_n) * 0.1
    
    # Reward for being centered and stable
    stability_reward = -abs(pos_x_n) * 0.05 - abs(angle_n) * 0.02
    
    # Combine all rewards
    reward = (
        center_approach * 2.0 +
        velocity_reduction * 1.5 +
        angle_reduction * 0.5 +
        ang_vel_reduction * 0.3 +
        ground_contact * 1.0 +
        height_reward +
        stability_reward -
        action_penalty
    )
    
    return reward