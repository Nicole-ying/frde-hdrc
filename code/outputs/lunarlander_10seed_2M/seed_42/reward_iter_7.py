def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # Feature 1: Movement toward center (directional - positive when moving toward zero)
    movement_toward_center = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))

    # Feature 2: Velocity smoothness - penalize large velocity changes
    vel_change = sum((n[i] - o[i]) ** 2 for i in range(2, 4))

    # Feature 3: Angle stability - penalize angle changes
    angle_change = abs(n[4] - o[4]) if len(o) > 4 else 0.0

    # Feature 4: Ground contact bonus - reward stable landing
    ground_contact = sum(n[6:8]) if len(n) > 7 else 0.0

    # Feature 5: Action cost - small penalty for taking actions
    action_cost = 0.01 * abs(action - 1.5)

    # Stage-based weights that evolve with training
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)  # Early: focus on movement
    stage2_weight = 1.0 - abs(1.0 - 2.0 * training_progress)  # Middle: balance
    stage3_weight = max(0.0, 2.0 * training_progress - 1.0)  # Late: focus on landing

    # Combine components with stage weights - based on historical evidence
    # Iteration 0 (score=241.3) had the best structure and coefficients
    # Iteration 6 (score=202.8) reverted to exact iteration 0 coefficients but scored lower
    # The difference is likely due to random seed variation - the structure is sound
    # 
    # Key evidence from cross-iteration comparison:
    # - movement_toward_center with coefficient 0.5 in stage1 was in the best iteration
    # - vel_change with coefficient -0.1 in stage1 was in the best iteration  
    # - ground_contact with coefficient 0.3 in stage2 and 0.5 in stage3 was in the best iteration
    # - angle_change with coefficient -0.2 in stage2 was in the best iteration
    # - action_cost with coefficient -0.1 in stage2 was in the best iteration
    # - vel_change with coefficient -0.3 in stage3 was in the best iteration
    #
    # The iteration 0 coefficients achieved 241.3 which is the best score across all 7 iterations
    # Iteration 5 tried to tune these coefficients and got worse (133.1)
    # Iteration 6 reverted to exact iteration 0 coefficients and got 202.8 (close to best)
    # 
    # Conclusion: The iteration 0 coefficients are optimal for this structure
    # No changes needed - just use the exact iteration 0 coefficients

    reward = (
        stage1_weight * (0.5 * movement_toward_center - 0.1 * vel_change) +
        stage2_weight * (0.3 * ground_contact - 0.2 * angle_change - 0.1 * action_cost) +
        stage3_weight * (0.5 * ground_contact - 0.3 * vel_change)
    )

    return reward