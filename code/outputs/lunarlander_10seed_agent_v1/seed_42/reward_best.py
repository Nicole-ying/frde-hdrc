def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement towards center - sum of absolute position changes
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Feature 2: Velocity stabilization - sum of squared velocity changes
    vel_change = sum((n[i] - o[i]) ** 2 for i in range(2, 4))
    
    # Feature 3: Angular stability - angle and angular velocity changes
    angle_change = abs(n[4]) - abs(o[4])
    ang_vel_change = abs(n[5]) - abs(o[5])
    
    # Feature 4: Ground contact bonus
    ground_contact = sum(n[6:8])
    
    # Feature 5: Action cost (small penalty for using engines)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.1
    elif action in [1, 3]:  # Side engines
        action_cost = 0.05
    
    # Stage-based weights that evolve with training progress
    # Stage 1 (early): Focus on reaching ground and stabilizing
    # Stage 2 (mid): Focus on precise landing
    # Stage 3 (late): Focus on efficiency
    
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)
    stage2_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    stage3_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Component rewards
    pos_reward = pos_change * 0.5
    vel_penalty = -vel_change * 0.1
    angle_reward = -angle_change * 0.3 - ang_vel_change * 0.2
    contact_reward = ground_contact * 0.5
    action_penalty = -action_cost
    
    # Combine with stage weights
    reward = (
        stage1_weight * (pos_reward + contact_reward * 0.3) +
        stage2_weight * (angle_reward + vel_penalty * 0.5 + contact_reward * 0.5) +
        stage3_weight * (vel_penalty * 0.7 + angle_reward * 0.3 + action_penalty * 0.2)
    )
    
    return float(reward)