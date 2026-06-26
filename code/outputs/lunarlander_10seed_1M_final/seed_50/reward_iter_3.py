def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays as generic lists for dimension-agnostic processing
    o = obs
    n = next_obs

    # Component 1: Movement magnitude change (encourages exploration/change)
    movement = sum(abs(n[i] - o[i]) for i in range(len(o)))

    # Component 2: State magnitude reduction (directional toward zero - encourages stability)
    state_mag_reduction = sum(abs(o[i]) for i in range(len(o))) - sum(abs(n[i]) for i in range(len(o)))

    # Component 3: Action penalty (discourages excessive actuation)
    action_cost = -0.01 * (1.0 if action != 0 else 0.0)

    # Component 4: Smoothness penalty (discourages jerky transitions)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o))) * 0.001

    # Component 5: Velocity penalty (directional - penalizes large absolute values)
    velocity_penalty = -sum(abs(n[i]) for i in range(len(o))) * 0.01

    # Component 6: Contact bonus (reward sustained ground contact)
    contact_bonus = 0.0
    for i in range(len(o)):
        if abs(o[i] - 1.0) < 0.05 and abs(n[i] - 1.0) < 0.05:
            contact_bonus += 0.3

    # Component 7: Survival bonus (reward staying alive)
    survival_bonus = 0.05

    # Stage-based weighting using training_progress (0.0 to 1.0)
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = max(0.0, 1.0 - abs(2.0 * training_progress - 1.0))
    late_weight = max(0.0, 2.0 * training_progress - 1.0)

    # Combine components with stage-appropriate weights
    # Early: explore with movement and light action cost, small survival bonus
    # Mid: start reducing state magnitude and penalizing velocity
    # Late: strongly reward stability, smoothness, and contact
    reward = (
        early_weight * (movement * 0.2 + action_cost + survival_bonus) +
        mid_weight * (state_mag_reduction * 0.3 + smoothness * 0.5 + velocity_penalty * 0.3 + contact_bonus * 0.2 + survival_bonus) +
        late_weight * (state_mag_reduction * 0.8 + smoothness * 1.0 + velocity_penalty * 0.8 + contact_bonus * 0.8 + survival_bonus)
    )

    return reward