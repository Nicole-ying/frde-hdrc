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

    # Combine components with stage weights
    reward = (
        stage1_weight * (0.8 * movement_toward_center - 0.1 * vel_magnitude - 0.2 * angle_penalty - 0.1 * angular_vel_penalty) +
        stage2_weight * (0.5 * ground_contact - 0.3 * angle_penalty - 0.2 * angular_vel_penalty - 0.1 * action_cost) +
        stage3_weight * (1.0 * ground_contact - 0.3 * vel_magnitude - 0.2 * angle_penalty - 0.2 * angular_vel_penalty)
    )

    return reward