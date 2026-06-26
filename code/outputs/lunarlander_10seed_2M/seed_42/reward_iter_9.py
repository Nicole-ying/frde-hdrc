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
    # - Iteration 6 (score=202.8) and Iteration 8 (score=198.6) used same coefficients but scored lower
    # - The consistent score range (198-241) with this structure suggests it's near-optimal
    # - Adding angle_penalty or angular_vel_penalty (Iterations 1-3) consistently hurt scores
    # - Tuning coefficients (Iteration 5, score=133.1) made things worse
    #
    # Key insight: The structure is sound but may benefit from a small survival bonus
    # to encourage longer episodes. The current structure has no explicit reward for
    # staying alive - it only rewards movement toward center and ground contact.
    # A small survival bonus (0.01 per step) would encourage the agent to avoid
    # early termination without dominating other signals.
    #
    # Adding a survival bonus of 0.01 per step gives +10 over 1000 steps, which is
    # small compared to movement_toward_center (which can be 2-4 per step) and
    # ground_contact (which gives 0.3-0.5 per step when active).

    survival_bonus = 0.01

    reward = (
        stage1_weight * (0.5 * movement_toward_center - 0.1 * vel_change + survival_bonus) +
        stage2_weight * (0.3 * ground_contact - 0.2 * angle_change - 0.1 * action_cost + survival_bonus) +
        stage3_weight * (0.5 * ground_contact - 0.3 * vel_change + survival_bonus)
    )

    return reward