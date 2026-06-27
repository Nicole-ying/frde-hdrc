def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large sudden changes)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Feature 4: Velocity change (from observation indices 2 and 3)
    vel_change = (abs(o[2]) - abs(n[2])) + (abs(o[3]) - abs(n[3]))
    
    # Feature 5: Ground contact bonus (from observation indices 6 and 7)
    ground_contact = n[6] + n[7]
    
    # Stage-based weights
    # Early training: focus on exploration and velocity reduction
    # Mid training: balance between position and velocity
    # Late training: focus on precision and ground contact
    
    if training_progress < 0.3:
        # Early stage: encourage movement and exploration
        w_abs = 1.0
        w_sq = -0.5
        w_action = -0.1
        w_vel = 0.5
        w_ground = 0.0
    elif training_progress < 0.7:
        # Mid stage: balance
        w_abs = 0.5
        w_sq = -0.3
        w_action = -0.05
        w_vel = 1.0
        w_ground = 0.5
    else:
        # Late stage: precision and landing
        w_abs = 0.2
        w_sq = -0.1
        w_action = -0.02
        w_vel = 2.0
        w_ground = 2.0
    
    # Combine components
    reward = (w_abs * abs_change + 
              w_sq * squared_change + 
              w_action * action_cost + 
              w_vel * vel_change + 
              w_ground * ground_contact)
    
    return reward