def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs
    
    # Position components (indices 0,1)
    pos_diff = abs(n[0]) - abs(o[0]) + abs(n[1]) - abs(o[1])
    
    # Velocity components (indices 2,3) - penalize velocity
    vel_penalty = (n[2] ** 2 + n[3] ** 2) * 0.1
    
    # Angle and angular velocity (indices 4,5) - penalize angle deviation and angular velocity
    angle_penalty = abs(n[4]) * 0.5 + abs(n[5]) * 0.2
    
    # Ground contact bonus (indices 6,7)
    contact_bonus = (n[6] + n[7]) * 0.5
    
    # Action penalty - small cost for using engines
    action_cost = 0.01 if action == 2 else 0.005 if action in [1, 3] else 0.0
    
    # Movement towards center
    movement_reward = (abs(o[0]) - abs(n[0])) * 0.3 + (abs(o[1]) - abs(n[1])) * 0.3
    
    # Stability bonus - penalize changes in velocity and angle
    stability_penalty = ((n[2] - o[2]) ** 2 + (n[3] - o[3]) ** 2) * 0.05 + (n[4] - o[4]) ** 2 * 0.1
    
    # Combine rewards
    reward = (
        movement_reward
        - vel_penalty
        - angle_penalty
        + contact_bonus
        - action_cost
        - stability_penalty
        + pos_diff * 0.1
    )
    
    return reward