def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Reward for moving towards center (horizontal position)
    pos_x_diff = abs(o[0]) - abs(n[0])
    
    # Reward for moving up (vertical position)
    pos_y_diff = n[1] - o[1]
    
    # Penalty for velocity (both horizontal and vertical)
    vel_x_penalty = abs(n[2])
    vel_y_penalty = abs(n[3])
    
    # Penalty for angle
    angle_penalty = abs(n[4])
    
    # Penalty for angular velocity
    ang_vel_penalty = abs(n[5])
    
    # Reward for ground contact
    ground_contact_reward = n[6] + n[7]
    
    # Small action penalty to encourage fuel efficiency
    action_penalty = 0.01 * (action == 2) + 0.005 * (action in [1, 3])
    
    # Combine all terms
    reward = (
        2.0 * pos_x_diff
        + 1.0 * pos_y_diff
        - 0.5 * vel_x_penalty
        - 1.0 * vel_y_penalty
        - 0.3 * angle_penalty
        - 0.1 * ang_vel_penalty
        + 2.0 * ground_contact_reward
        - action_penalty
    )
    
    return reward