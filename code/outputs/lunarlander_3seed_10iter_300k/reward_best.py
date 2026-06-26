def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Dimension-agnostic transition features

    # Movement magnitude - reward moving toward target (change in absolute values)
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Velocity change bonus - reward consistent, smooth state transitions
    # This provides denser feedback than just movement
    velocity_change = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # Action penalty (small cost for taking actions)
    action_cost = 0.01 * action

    # Survival bonus - reward staying alive longer
    survival_bonus = 0.1

    # Stage weights based on training progress
    # Early: focus on movement and exploration
    # Late: focus on stability and survival
    movement_weight = 1.0 - 0.3 * training_progress
    velocity_weight = 0.5 * (1.0 - 0.5 * training_progress)
    action_weight = 0.05 * (1.0 - 0.5 * training_progress)
    survival_weight = 0.1 * training_progress

    # Combine components
    reward = (
        movement_weight * movement
        + velocity_weight * velocity_change
        - action_weight * action_cost
        + survival_weight * survival_bonus
    )

    return float(reward)