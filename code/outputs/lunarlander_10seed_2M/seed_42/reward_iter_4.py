def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # Feature 1: Movement toward center (directional - positive when moving toward zero)
    movement_toward_center = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))

    # Feature 2: Velocity magnitude penalty (discourage high speed)
    vel_magnitude = sum(abs(n[i]) for i in range(2, 4))

    # Feature 3: Angle stability - penalize absolute angle (not change)
    angle_penalty = abs(n[4]) if len(n) > 4 else 0.0

    # Feature 4: Angular velocity penalty - discourage spinning
    angular_vel_penalty = abs(n[5]) if len(n) > 5 else 0.0

    # Feature 5: Ground contact bonus - reward stable landing
    ground_contact = sum(n[6:8]) if len(n) > 7 else 0.0

    # Feature 6: Action cost - small penalty for using engines
    action_cost = 0.01 * abs(action - 1.5)

    # Stage-based weights that evolve with training
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)  # Early: focus on movement
    stage2_weight = 1.0 - abs(1.0 - 2.0 * training_progress)  # Middle: balance
    stage3_weight = max(0.0, 2.0 * training_progress - 1.0)  # Late: focus on landing

    # Build reward based on best-performing iteration 0 structure with tuned coefficients
    # Key evidence from history:
    # - Iteration 0 (score=241.3) had: movement_toward_center, vel_change, angle_change, ground_contact, action_cost
    # - Iteration 1 (score=-109.4) added angle_penalty, angular_vel_penalty with high weights - catastrophic
    # - Iteration 2 (score=105.9) kept angle_penalty, angular_vel_penalty but reduced weights - still worse than iter 0
    # - Iteration 3 (score=154.4) removed angle_penalty/angular_vel_penalty from stage1, reduced ground_contact - improved but still below iter 0
    # 
    # Conclusion: angle_penalty and angular_vel_penalty are harmful when applied early (stage1)
    # Ground_contact is important but was over-emphasized in iter 2
    # Movement_toward_center is the strongest positive signal
    # Vel_change (smoothness) from iter 0 was better than vel_magnitude (absolute speed)
    
    # Use vel_change instead of vel_magnitude - it was in the best iteration
    vel_change = sum((n[i] - o[i]) ** 2 for i in range(2, 4))
    
    # Use angle_change instead of angle_penalty - it was in the best iteration
    angle_change = abs(n[4] - o[4]) if len(o) > 4 else 0.0

    # Combine components with stage weights - based on iteration 0 structure
    reward = (
        stage1_weight * (0.5 * movement_toward_center - 0.1 * vel_change) +
        stage2_weight * (0.3 * ground_contact - 0.2 * angle_change - 0.1 * action_cost) +
        stage3_weight * (0.5 * ground_contact - 0.3 * vel_change)
    )

    return reward