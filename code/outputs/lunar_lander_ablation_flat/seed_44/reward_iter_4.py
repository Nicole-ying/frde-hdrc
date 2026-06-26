def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Reward for moving towards landing pad (reducing absolute position)
    position_improvement = sum(abs(o[i]) - abs(n[i]) for i in range(2))
    
    # Reward for reducing velocity (slowing down)
    velocity_reduction = sum(abs(o[i]) - abs(n[i]) for i in range(2, 4))
    
    # Reward for reducing angle and angular velocity (stabilizing)
    angle_improvement = abs(o[4]) - abs(n[4])
    angular_vel_reduction = abs(o[5]) - abs(n[5])
    
    # Reward for ground contact (landing)
    ground_contact = sum(n[6:8])
    
    # Small penalty for using actions (fuel efficiency)
    action_penalty = 0.01 * (1 if action == 2 else 0)
    
    # Compute reward
    reward = (
        2.0 * position_improvement +
        1.5 * velocity_reduction +
        1.0 * angle_improvement +
        0.5 * angular_vel_reduction +
        10.0 * ground_contact -
        action_penalty
    )
    
    return reward