def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs
    
    # Position error: encourage moving toward center (x=0) and landing pad height
    # obs[0] is normalized x position, obs[1] is normalized y position relative to helipad
    pos_error = abs(o[0]) - abs(n[0]) + abs(o[1]) - abs(n[1])
    
    # Velocity reduction: encourage slowing down
    # obs[2] is x velocity, obs[3] is y velocity
    vel_penalty = abs(o[2]) - abs(n[2]) + abs(o[3]) - abs(n[3])
    
    # Angle stabilization: encourage upright orientation
    # obs[4] is angle, obs[5] is angular velocity
    angle_improvement = abs(o[4]) - abs(n[4])
    ang_vel_improvement = abs(o[5]) - abs(n[5])
    
    # Ground contact bonus: encourage landing
    # obs[6] and obs[7] are leg contact indicators
    contact_bonus = (n[6] + n[7]) - (o[6] + o[7])
    
    # Action penalty: discourage unnecessary engine use
    # action 2 is main engine, action 1 and 3 are side engines
    action_cost = 0.0
    if action == 2:
        action_cost = -0.1
    elif action == 1 or action == 3:
        action_cost = -0.05
    
    # Smoothness penalty: discourage large changes in state
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(6)) * 0.01
    
    # Combine rewards with fixed coefficients
    reward = (
        2.0 * pos_error +
        1.5 * vel_penalty +
        1.0 * angle_improvement +
        0.5 * ang_vel_improvement +
        3.0 * contact_bonus +
        action_cost +
        smoothness
    )
    
    return reward