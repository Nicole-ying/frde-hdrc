def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement magnitude (change in absolute values)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (captures large deviations)
    sq_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (discourage excessive action usage)
    action_cost = 0.0
    if action in [1, 2, 3]:
        action_cost = 0.01
    
    # Feature 4: Stability signal (penalize large angular changes)
    # Assuming angle is at index 4 and angular velocity at index 5
    angle_change = abs(n[4] - o[4]) if len(o) > 4 else 0.0
    angular_vel_change = abs(n[5] - o[5]) if len(o) > 5 else 0.0
    stability_penalty = angle_change + 0.5 * angular_vel_change
    
    # Feature 5: Ground contact bonus (indices 6 and 7)
    ground_contact = 0.0
    if len(o) > 6:
        ground_contact = n[6] + n[7]  # Sum of leg contact indicators
    
    # Stage-based weights that evolve with training progress
    # Stage 1 (early): Focus on exploration and movement
    w_abs_diff = 0.5 + 0.5 * (1.0 - training_progress)
    # Stage 2 (mid): Focus on stability and precision
    w_sq_diff = 0.3 + 0.7 * min(1.0, training_progress * 2.0)
    # Stage 3 (late): Focus on landing and contact
    w_ground = 0.0 + 1.0 * max(0.0, min(1.0, (training_progress - 0.5) * 2.0))
    
    # Constant weights
    w_action = 0.1
    w_stability = 0.2 + 0.3 * training_progress
    
    # Compute reward components
    reward_abs = w_abs_diff * abs_diff
    reward_sq = -w_sq_diff * sq_diff  # Penalize large squared changes
    reward_action = -w_action * action_cost
    reward_stability = -w_stability * stability_penalty
    reward_ground = w_ground * ground_contact
    
    # Combine components
    reward = reward_abs + reward_sq + reward_action + reward_stability + reward_ground
    
    return reward