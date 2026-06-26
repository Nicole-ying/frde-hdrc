def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state components from observation
    o = obs
    n = next_obs
    
    # Position components (indices 0,1)
    pos_x = o[0]
    pos_y = o[1]
    next_pos_x = n[0]
    next_pos_y = n[1]
    
    # Velocity components (indices 2,3)
    vel_x = o[2]
    vel_y = o[3]
    next_vel_x = n[2]
    next_vel_y = n[3]
    
    # Angle and angular velocity (indices 4,5)
    angle = o[4]
    ang_vel = o[5]
    next_angle = n[4]
    next_ang_vel = n[5]
    
    # Ground contact (indices 6,7)
    leg1_contact = o[6]
    leg2_contact = o[7]
    next_leg1_contact = n[6]
    next_leg2_contact = n[7]
    
    # Reward for being close to landing pad (center at x=0, y=0)
    distance_penalty = -0.01 * (abs(pos_x) + abs(pos_y))
    next_distance_penalty = -0.01 * (abs(next_pos_x) + abs(next_pos_y))
    distance_reward = next_distance_penalty - distance_penalty
    
    # Reward for reducing velocity (especially vertical)
    speed_reduction = -0.05 * (abs(next_vel_y) - abs(vel_y))
    
    # Reward for being upright (angle close to 0)
    angle_penalty = -0.02 * abs(angle)
    next_angle_penalty = -0.02 * abs(next_angle)
    angle_reward = next_angle_penalty - angle_penalty
    
    # Reward for reducing angular velocity
    ang_vel_reduction = -0.01 * (abs(next_ang_vel) - abs(ang_vel))
    
    # Reward for ground contact (landing)
    ground_contact_reward = 0.5 * (next_leg1_contact + next_leg2_contact)
    
    # Action penalty (encourage fuel efficiency)
    action_penalty = -0.01 * (1.0 if action == 2 else 0.5 if action in [1, 3] else 0.0)
    
    # Small bonus for being near the landing area with low velocity
    landing_bonus = 0.0
    if abs(next_pos_x) < 0.1 and abs(next_pos_y) < 0.1 and abs(next_vel_y) < 0.05:
        landing_bonus = 1.0
    
    # Combine all reward components
    reward = (distance_reward + speed_reduction + angle_reward + 
              ang_vel_reduction + ground_contact_reward + action_penalty + landing_bonus)
    
    return reward