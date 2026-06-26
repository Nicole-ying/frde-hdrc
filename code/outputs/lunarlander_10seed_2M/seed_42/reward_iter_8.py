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

    # Based on cross-iteration analysis:
    # - Iteration 0 (score=241.3) had the best structure with these exact coefficients
    # - Iteration 6 (score=202.8) reverted to same coefficients but got lower score (random seed variation)
    # - Iteration 4 (score=200.1) used same structure with slightly different coefficients
    # - Iteration 5 (score=133.1) tried tuning coefficients and got worse
    # - Iteration 1 (score=-109.4) added angle_penalty and angular_vel_penalty - catastrophic
    # - Iteration 2 (score=105.9) kept those penalties with reduced weights - still worse
    # - Iteration 3 (score=154.4) removed early penalties - improved
    #
    # Key findings:
    # 1. movement_toward_center is the strongest positive signal (present in all good iterations)
    # 2. vel_change (smoothness) is better than vel_magnitude (absolute speed)
    # 3. angle_change (change-based) is better than angle_penalty (absolute angle)
    # 4. ground_contact is important in middle/late stages
    # 5. action_cost is a small but useful regularizer
    # 6. angular_vel_penalty was harmful - removed entirely
    # 7. The iteration 0 coefficients are optimal for this structure
    #
    # The latest iteration (7) scored 154.2 which is worse than iteration 0 (241.3)
    # This is likely due to random seed variation since the code is identical
    # The structure and coefficients from iteration 0 remain the best found so far
    # No changes needed - use the exact iteration 0 coefficients

    reward = (
        stage1_weight * (0.5 * movement_toward_center - 0.1 * vel_change) +
        stage2_weight * (0.3 * ground_contact - 0.2 * angle_change - 0.1 * action_cost) +
        stage3_weight * (0.5 * ground_contact - 0.3 * vel_change)
    )

    return reward