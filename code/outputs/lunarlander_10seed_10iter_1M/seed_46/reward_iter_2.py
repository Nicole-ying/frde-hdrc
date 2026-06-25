def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Component 1: Movement towards stability - directional signal toward zero
    # Using absolute difference: positive when moving toward zero
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.15

    # Component 2: Smoothness penalty - penalize large changes
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_change * 0.05

    # Component 3: Action cost - penalize engine usage
    action_cost = -0.01 * (1 if action != 0 else 0)

    # Component 4: Contact bonus - reward ground contact
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = (o[6] + o[7]) * 0.5

    # Component 5: Velocity penalty - penalize high speed (from obs indices 2,3)
    velocity_penalty = 0.0
    if len(o) >= 4:
        vel_x = o[2]
        vel_y = o[3]
        velocity_penalty = -0.05 * (vel_x**2 + vel_y**2)

    # Component 6: Angle stability - penalize large angles (from obs index 4)
    angle_penalty = 0.0
    if len(o) >= 5:
        angle = o[4]
        angle_penalty = -0.1 * abs(angle)

    # Stage weights based on training_progress
    # Early: focus on movement and stability
    # Middle: balance all components
    # Late: focus on contact and smooth landing
    stage1_weight = max(0.0, 1.0 - training_progress * 2.0)
    stage2_weight = min(1.0, training_progress * 2.0) * (1.0 - max(0.0, training_progress * 2.0 - 1.0))
    stage3_weight = max(0.0, training_progress * 2.0 - 1.0)

    # Combine components with stage weights
    reward = (
        stage1_weight * (movement_reward + angle_penalty + velocity_penalty) +
        stage2_weight * (smoothness_penalty + action_cost + movement_reward * 0.5) +
        stage3_weight * (contact_bonus + smoothness_penalty * 0.5)
    )

    return reward