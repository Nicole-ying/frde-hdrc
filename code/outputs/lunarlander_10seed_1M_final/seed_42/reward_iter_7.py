def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Absolute value reduction - encourages moving towards zero in normalized space
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change magnitude - captures overall movement
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost - penalize engine usage
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # Feature 4: Ground contact bonus (from last two obs dimensions)
    ground_contact = n[6] + n[7]  # Sum of leg contact indicators
    
    # Feature 5: Velocity magnitude change (from obs indices 2,3)
    vel_change = abs(n[2]) + abs(n[3]) - (abs(o[2]) + abs(o[3]))
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on reducing absolute values and exploring
    w_abs = 1.0 - 0.5 * training_progress
    # Middle stage: focus on smooth movement and ground contact
    w_sq = 0.5 + 0.5 * training_progress
    # Late stage: reduce action cost and emphasize ground contact
    w_action = 0.3 + 0.7 * training_progress
    w_contact = 0.0 + 1.0 * training_progress
    # Velocity change weight decreases over time
    w_vel = 1.0 - 0.8 * training_progress
    
    # Combine components with stage weights
    reward = (
        w_abs * abs_diff * 0.1 +
        w_sq * (-sq_change) * 0.05 +
        w_action * (-action_cost) * 0.2 +
        w_contact * ground_contact * 0.5 +
        w_vel * (-vel_change) * 0.1
    )
    
    return reward