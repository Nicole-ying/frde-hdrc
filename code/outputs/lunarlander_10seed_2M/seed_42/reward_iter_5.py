def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # Feature 1: Movement toward center (directional - positive when moving toward zero)
    movement_toward_center = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))

    # Feature 2: Velocity smoothness - penalize large velocity changes (from best iteration 0)
    vel_change = sum((n[i] - o[i]) ** 2 for i in range(2, 4))

    # Feature 3: Angle stability - penalize angle changes (from best iteration 0)
    angle_change = abs(n[4] - o[4]) if len(o) > 4 else 0.0

    # Feature 4: Ground contact bonus - reward stable landing
    ground_contact = sum(n[6:8]) if len(n) > 7 else 0.0

    # Feature 5: Action cost - small penalty for taking actions
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
    # - Iteration 4 (score=200.1) reverted to iteration 0 structure with vel_change and angle_change - improved but still below iter 0
    #
    # Conclusion: 
    # - movement_toward_center is the strongest positive signal (present in all good iterations)
    # - vel_change (smoothness) is better than vel_magnitude (absolute speed) - iter 0 vs iter 3 comparison
    # - angle_change (change-based) is better than angle_penalty (absolute) - iter 0 vs iter 1/2 comparison
    # - ground_contact is important but was over-emphasized in iter 2
    # - action_cost is a small but useful regularizer
    # - angular_vel_penalty was harmful - removed entirely
    #
    # Tuning: Increase movement_toward_center weight, decrease vel_change penalty, 
    # keep ground_contact moderate, keep angle_change moderate

    # Combine components with stage weights - based on iteration 0 structure with tuned coefficients
    reward = (
        stage1_weight * (0.6 * movement_toward_center - 0.08 * vel_change) +
        stage2_weight * (0.35 * ground_contact - 0.15 * angle_change - 0.08 * action_cost) +
        stage3_weight * (0.55 * ground_contact - 0.25 * vel_change)
    )

    return reward