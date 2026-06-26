def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Component 1: Movement toward stability (reducing absolute values)
    # Directional signal: reward moving toward zero for all state dimensions
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_change * 0.3

    # Component 2: Smoothness of transition - penalize large jumps
    diff_abs = sum(abs(n[i] - o[i]) for i in range(len(o)))
    smoothness_penalty = -diff_abs * 0.05

    # Component 3: Action cost - penalize large actions
    action_cost = -0.01 * float(action)

    # Component 4: Contact bonus - reward maintaining ground contact
    contact_bonus = 0.0
    if len(o) >= 8:
        leg1_contact = o[6]
        leg2_contact = o[7]
        contact_bonus = (leg1_contact + leg2_contact) * 0.3

    # Component 5: Velocity penalty - discourage high speeds
    if len(o) >= 4:
        vel_mag = abs(o[2]) + abs(o[3])
        velocity_penalty = -vel_mag * 0.05
    else:
        velocity_penalty = 0.0

    # Stage weights based on training_progress
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(training_progress - 0.5) * 2.0
    late_weight = max(0.0, 2.0 * training_progress - 1.0)

    # Combine components with stage-adaptive weights
    reward = (
        early_weight * movement_reward +
        mid_weight * (smoothness_penalty + velocity_penalty) +
        late_weight * contact_bonus +
        action_cost * (1.0 - training_progress * 0.5)
    )

    return reward