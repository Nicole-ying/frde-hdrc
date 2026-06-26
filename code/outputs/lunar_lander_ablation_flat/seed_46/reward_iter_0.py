def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Position change (indices 0,1 are position)
    pos_change = abs(n[0]) - abs(o[0]) + abs(n[1]) - abs(o[1])
    
    # Velocity change (indices 2,3 are velocity)
    vel_change = abs(n[2]) - abs(o[2]) + abs(n[3]) - abs(o[3])
    
    # Angle change (index 4 is angle)
    angle_change = abs(n[4]) - abs(o[4])
    
    # Angular velocity change (index 5)
    angvel_change = abs(n[5]) - abs(o[5])
    
    # Ground contact (indices 6,7)
    contact_gain = (n[6] - o[6]) + (n[7] - o[7])
    
    # Action penalty (discrete 0-3)
    action_cost = 0.01 * (action if action < 4 else 0)
    
    # Reward components with fixed coefficients
    reward = (
        -0.5 * pos_change
        -0.3 * vel_change
        -0.2 * angle_change
        -0.1 * angvel_change
        + 2.0 * contact_gain
        - action_cost
    )
    
    return reward