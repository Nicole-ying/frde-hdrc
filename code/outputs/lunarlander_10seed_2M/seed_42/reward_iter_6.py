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

    # Combine components with stage weights
    # Based on historical analysis:
    # - Iteration 0 (score=241.3) had the best structure: movement_toward_center, vel_change, angle_change, ground_contact, action_cost
    # - Iteration 4 (score=200.1) reverted to this structure and improved from 154.4
    # - Iteration 5 (score=133.1) tried to tune coefficients but got worse - the coefficients from iter 0 were actually better
    # - Key insight: movement_toward_center is the strongest signal, ground_contact is important in middle/late stages
    # - vel_change and angle_change are useful penalties but should be moderate
    # - action_cost is a small regularizer
    # 
    # The failure in iteration 5 was over-tuning: increasing movement_toward_center to 0.6 and decreasing vel_change to 0.08
    # made the agent too aggressive. Reverting to iteration 0's exact coefficients which achieved 241.3.
    
    reward = (
        stage1_weight * (0.5 * movement_toward_center - 0.1 * vel_change) +
        stage2_weight * (0.3 * ground_contact - 0.2 * angle_change - 0.1 * action_cost) +
        stage3_weight * (0.5 * ground_contact - 0.3 * vel_change)
    )

    return reward