def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract arrays for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Reward for moving towards target (positive change in position components)
    # obs[0] and obs[1] are normalized position, next_obs[0] and next_obs[1] are new position
    # We want to move towards center (0,0) for landing pad
    pos_change = (abs(o[0]) - abs(n[0])) + (abs(o[1]) - abs(n[1]))
    
    # Reward for reducing velocity (smooth landing)
    # obs[2] and obs[3] are velocity components
    vel_change = (abs(o[2]) - abs(n[2])) + (abs(o[3]) - abs(n[3]))
    
    # Reward for upright orientation (angle close to 0)
    # obs[4] is angle, we want it to decrease
    angle_change = abs(o[4]) - abs(n[4])
    
    # Reward for reducing angular velocity
    # obs[5] is angular velocity
    angvel_change = abs(o[5]) - abs(n[5])
    
    # Penalty for using main engine (action 2)
    main_engine_penalty = -0.1 if action == 2 else 0.0
    
    # Penalty for using side engines (action 1 or 3)
    side_engine_penalty = -0.05 if action in [1, 3] else 0.0
    
    # Reward for ground contact (legs touching)
    # obs[6] and obs[7] are leg contact indicators
    ground_contact_reward = 1.0 if (n[6] > 0.5 or n[7] > 0.5) else 0.0
    
    # Combine all rewards with fixed coefficients
    reward = (
        2.0 * pos_change +
        1.5 * vel_change +
        1.0 * angle_change +
        0.5 * angvel_change +
        main_engine_penalty +
        side_engine_penalty +
        10.0 * ground_contact_reward
    )
    
    return reward