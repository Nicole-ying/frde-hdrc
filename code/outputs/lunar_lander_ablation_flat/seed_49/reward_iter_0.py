def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Extract position and velocity components (indices 0-3)
    pos_diff = abs(o[0]) - abs(n[0]) + abs(o[1]) - abs(n[1])
    vel_diff = abs(o[2]) - abs(n[2]) + abs(o[3]) - abs(n[3])
    
    # Extract angle and angular velocity (indices 4-5)
    angle_diff = abs(o[4]) - abs(n[4])
    angvel_diff = abs(o[5]) - abs(n[5])
    
    # Ground contact signals (indices 6-7)
    contact_gain = n[6] + n[7]
    
    # Movement penalty (squared velocity change)
    vel_change = (n[2] - o[2]) ** 2 + (n[3] - o[3]) ** 2
    
    # Action penalty (encourage efficient control)
    action_cost = 0.01 * (action == 2) + 0.005 * (action in [1, 3])
    
    # Reward components with fixed coefficients
    position_reward = 0.5 * pos_diff
    velocity_reward = 0.3 * vel_diff
    angle_reward = 0.2 * angle_diff
    angvel_reward = 0.1 * angvel_diff
    contact_bonus = 1.0 * contact_gain
    movement_penalty = -0.1 * vel_change
    
    reward = (position_reward + velocity_reward + angle_reward + 
              angvel_reward + contact_bonus + movement_penalty - action_cost)
    
    return reward