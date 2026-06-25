def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs

    # Dimension-agnostic transition features

    # 1. Movement toward center (encourage staying near origin)
    # Use sum of absolute values as a proxy for distance from origin
    dist_old = sum(abs(o[i]) for i in range(len(o)))
    dist_new = sum(abs(n[i]) for i in range(len(n)))
    movement_reward = (dist_old - dist_new) * 0.3

    # 2. Smoothness penalty - penalize large changes between steps
    change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -change * 0.05

    # 3. Action penalty - discourage excessive action usage
    action_cost = -0.05 * float(action)

    # 4. Contact bonus - reward sustained contact (staying on ground)
    contact_bonus = 0.0
    if len(o) >= 8:
        new_contacts = n[6] + n[7]
        contact_bonus = new_contacts * 1.0

    # 5. Survival bonus - reward staying alive longer
    survival_bonus = 0.2

    # 6. Velocity penalty - penalize high speed (crashes)
    # Indices 2 and 3 are velocity components
    vel_penalty = 0.0
    if len(o) >= 4:
        vel_x = abs(n[2])
        vel_y = abs(n[3])
        vel_penalty = -0.2 * (vel_x + vel_y)

    # 7. Angular velocity penalty - penalize spinning (index 5)
    angular_penalty = 0.0
    if len(o) >= 6:
        angular_penalty = -0.1 * abs(n[5])

    # Combine with stage weights based on training_progress
    # Early training: focus on survival and movement
    # Late training: focus on smoothness and precision
    early_weight = 1.0 - training_progress
    late_weight = training_progress

    reward = (
        early_weight * (movement_reward + survival_bonus + action_cost + contact_bonus) +
        late_weight * (smoothness_penalty + vel_penalty + angular_penalty)
    )

    return reward