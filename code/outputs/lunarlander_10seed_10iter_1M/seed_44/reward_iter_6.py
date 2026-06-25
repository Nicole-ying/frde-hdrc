def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Change in absolute values (encourages moving towards beneficial states)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change magnitude (captures overall movement)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small cost to discourage excessive action)
    action_cost = 0.01 * action
    
    # Feature 4: Velocity-related signal from observation indices 2 and 3
    vel_change = abs(o[2]) - abs(n[2]) + abs(o[3]) - abs(n[3])
    
    # Feature 5: Ground contact signal from observation indices 6 and 7
    ground_contact = n[6] + n[7]
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on exploration and movement
    w_abs = 0.5 + 0.3 * training_progress
    w_sq = 0.3 - 0.2 * training_progress
    w_action = -0.02 + 0.01 * training_progress
    w_vel = 0.4 + 0.2 * training_progress
    w_ground = 0.1 + 0.5 * training_progress
    
    # Combine components
    reward = (
        w_abs * abs_change +
        w_sq * sq_change +
        w_action * action_cost +
        w_vel * vel_change +
        w_ground * ground_contact
    )
    
    return reward