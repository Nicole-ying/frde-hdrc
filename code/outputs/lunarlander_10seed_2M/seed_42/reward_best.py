def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement towards center - penalize moving away from zero
    movement_toward_center = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))
    
    # Feature 2: Velocity change - penalize large velocity changes
    vel_change = sum((n[i] - o[i]) ** 2 for i in range(2, 4))
    
    # Feature 3: Angle stability - penalize angle changes
    angle_change = abs(n[4] - o[4]) if len(o) > 4 else 0.0
    
    # Feature 4: Ground contact bonus
    ground_contact = sum(n[6:8]) if len(n) > 7 else 0.0
    
    # Feature 5: Action cost - small penalty for taking actions
    action_cost = 0.01 * abs(action - 1.5)  # Discrete action penalty
    
    # Stage-based weights that evolve with training
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)  # Early: focus on movement
    stage2_weight = 1.0 - abs(1.0 - 2.0 * training_progress)  # Middle: balance
    stage3_weight = max(0.0, 2.0 * training_progress - 1.0)  # Late: focus on landing
    
    # Combine components with stage weights
    reward = (
        stage1_weight * (0.5 * movement_toward_center - 0.1 * vel_change) +
        stage2_weight * (0.3 * ground_contact - 0.2 * angle_change - 0.1 * action_cost) +
        stage3_weight * (0.5 * ground_contact - 0.3 * vel_change)
    )
    
    return reward