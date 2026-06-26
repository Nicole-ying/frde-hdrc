def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs
    
    # Position components (indices 0,1)
    # Velocity components (indices 2,3)
    # Angle and angular velocity (indices 4,5)
    # Ground contact flags (indices 6,7)
    
    # Reward for moving toward landing pad (negative x means moving left, positive x means moving right)
    # The landing pad is at x=0, so we want to minimize absolute x position
    pos_x_diff = abs(o[0]) - abs(n[0])
    
    # Reward for descending (negative y means going down in this coordinate system)
    # We want the lander to go down (negative velocity in y)
    vel_y_change = n[3] - o[3]  # positive means slowing descent or going up
    
    # Reward for reducing velocity magnitude (safer landing)
    vel_mag_old = (o[2]**2 + o[3]**2)**0.5
    vel_mag_new = (n[2]**2 + n[3]**2)**0.5
    vel_reduction = vel_mag_old - vel_mag_new
    
    # Reward for upright orientation (angle close to 0)
    angle_penalty = -abs(n[4])
    
    # Reward for reducing angular velocity
    angvel_reduction = abs(o[5]) - abs(n[5])
    
    # Reward for ground contact (landing legs touching)
    ground_contact_reward = n[6] + n[7]
    
    # Action penalty (encourage efficient use of engines)
    # action 0: do nothing, action 1: left, action 2: main, action 3: right
    action_cost = 0.0
    if action == 2:  # main engine
        action_cost = -0.1
    elif action == 1 or action == 3:  # side engines
        action_cost = -0.05
    
    # Combine rewards with fixed coefficients
    reward = (
        0.5 * pos_x_diff +          # Move toward center
        0.3 * vel_y_change +        # Control descent rate
        0.4 * vel_reduction +       # Reduce speed
        -0.2 * angle_penalty +      # Stay upright
        0.1 * angvel_reduction +    # Reduce spin
        2.0 * ground_contact_reward + # Touch ground with legs
        action_cost                 # Engine usage cost
    )
    
    return reward