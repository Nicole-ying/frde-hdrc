def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state components
    o = obs
    n = next_obs
    
    # Position and velocity components (indices 0-3)
    # Position error reduction: encourage moving toward center
    pos_error_reduction = abs(o[0]) - abs(n[0]) + abs(o[1]) - abs(n[1])
    
    # Velocity reduction: encourage slowing down
    vel_reduction = abs(o[2]) - abs(n[2]) + abs(o[3]) - abs(n[3])
    
    # Angular stability: encourage upright orientation and low angular velocity
    angle_improvement = abs(o[4]) - abs(n[4])
    ang_vel_reduction = abs(o[5]) - abs(n[5])
    
    # Ground contact bonus: reward both legs touching ground
    leg_contact_bonus = n[6] + n[7]
    
    # Action penalty: discourage unnecessary engine usage
    # action 2 = main engine, action 1 or 3 = side engines
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = -0.1
    elif action == 1 or action == 3:  # Side engines
        action_cost = -0.05
    
    # Smoothness: penalize large changes in position/velocity
    smoothness_penalty = -0.01 * ((n[0] - o[0])**2 + (n[1] - o[1])**2 + (n[2] - o[2])**2 + (n[3] - o[3])**2)
    
    # Combine all signals with fixed coefficients
    reward = (
        2.0 * pos_error_reduction +
        1.5 * vel_reduction +
        1.0 * angle_improvement +
        0.5 * ang_vel_reduction +
        1.0 * leg_contact_bonus +
        action_cost +
        smoothness_penalty
    )
    
    return reward