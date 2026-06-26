def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Movement penalty - encourage moving towards landing (positive y direction)
    movement_penalty = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Velocity change - encourage reducing velocity
    velocity_change = sum((n[i] - o[i]) ** 2 for i in range(2, 4))
    
    # Angle penalty - encourage upright position
    angle_penalty = abs(n[4]) * 0.5
    
    # Ground contact bonus
    ground_bonus = n[6] + n[7]
    
    # Action penalty - small cost for using engines
    action_cost = 0.01 * (1 if action == 2 else 0.5 if action in [1, 3] else 0.0)
    
    # Height penalty - encourage being closer to ground
    height_penalty = abs(n[1]) * 0.1
    
    reward = movement_penalty * 0.5 - velocity_change * 0.3 - angle_penalty + ground_bonus * 2.0 - action_cost - height_penalty
    
    return reward