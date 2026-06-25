def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # ===== SIGNAL CATEGORIES =====

    # 1. TRANSITION SIGNAL: movement toward desirable states (directional)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # 2. SMOOTHNESS SIGNAL: penalize large jumps between states
    smoothness = -sum(abs(n[i] - o[i]) for i in range(len(o)))

    # 3. STABILITY SIGNAL: penalize angular velocity and angle (prevent tumbling)
    angle_penalty = 0.0
    angvel_penalty = 0.0
    if len(o) >= 6:
        angle_penalty = -abs(o[4]) * 0.25
        angvel_penalty = -abs(o[5]) * 0.15

    # 4. VELOCITY SIGNAL: penalize high velocity (soft landing)
    velocity_penalty = 0.0
    if len(o) >= 4:
        velocity_penalty = -((o[2]**2 + o[3]**2) ** 0.5) * 0.12

    # 5. CONTACT/LANDING SIGNAL: reward ground contact
    ground_contact = 0.0
    if len(o) >= 8:
        ground_contact = o[6] + o[7]

    # 6. ACTION COST: penalize engine usage (efficiency)
    action_cost = 0.0
    if action == 2:
        action_cost = -0.2
    elif action in [1, 3]:
        action_cost = -0.08

    # ===== STAGE WEIGHTS =====
    early_weight = 1.0 - training_progress
    middle_weight = 4.0 * training_progress * (1.0 - training_progress)
    late_weight = training_progress

    # ===== COMBINE COMPONENTS =====
    # Based on historical analysis:
    # - Iteration 4 (best score 160.9) used reduced penalties and higher survival bonus
    # - Iteration 5 (score 158.7) had slightly worse performance, suggesting need for fine-tuning
    # - Key insight: reduce penalty magnitudes further, increase ground contact reward, 
    #   and add a small velocity-based bonus for moving toward the landing pad
    reward = (
        early_weight * (abs_change * 0.6 + angle_penalty * 0.15 + angvel_penalty * 0.1) +
        middle_weight * (smoothness * 0.02 + velocity_penalty * 0.05) +
        late_weight * (ground_contact * 2.0 + action_cost * 0.15) +
        0.2  # Survival bonus to encourage longer episodes
    )

    return reward