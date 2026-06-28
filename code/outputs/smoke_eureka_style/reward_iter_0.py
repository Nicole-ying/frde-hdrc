def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Extract relevant observations
    # o[0], o[1]: x, y position
    # o[2], o[3]: x, y velocity
    # o[4]: angle
    # o[5]: angular velocity
    # o[6], o[7]: leg contact flags
    
    # Movement reward - encourage exploration early, precision later
    movement_magnitude = abs(n[2]) + abs(n[3]) + abs(n[5])
    if training_progress < 0.3:
        movement_weight = 0.3
        movement_reward = movement_weight * movement_magnitude
    elif training_progress < 0.7:
        movement_weight = 0.1
        movement_reward = movement_weight * movement_magnitude
    else:
        movement_weight = -0.2
        movement_reward = movement_weight * movement_magnitude
    
    # Stability reward - penalize rapid changes in late training
    vel_change = abs(n[2] - o[2]) + abs(n[3] - o[3])
    ang_vel_change = abs(n[5] - o[5])
    stability_penalty = vel_change + ang_vel_change
    
    if training_progress < 0.3:
        stability_weight = -0.05
    elif training_progress < 0.7:
        stability_weight = -0.15
    else:
        stability_weight = -0.4
    
    stability_reward = stability_weight * stability_penalty
    
    # Contact reward - encourage ground contact in late training
    leg_contact = n[6] + n[7]
    if training_progress < 0.3:
        contact_weight = 0.1
    elif training_progress < 0.7:
        contact_weight = 0.3
    else:
        contact_weight = 0.6
    
    contact_reward = contact_weight * leg_contact
    
    # Position-based reward - encourage staying near center
    pos_distance = abs(n[0]) + abs(n[1])
    if training_progress < 0.3:
        pos_weight = -0.1
    elif training_progress < 0.7:
        pos_weight = -0.2
    else:
        pos_weight = -0.3
    
    position_reward = pos_weight * pos_distance
    
    # Action penalty - discourage excessive action usage
    action_penalty = -0.01 * action
    
    total_reward = movement_reward + stability_reward + contact_reward + position_reward + action_penalty
    
    return total_reward