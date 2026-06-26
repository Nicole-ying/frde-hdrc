def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Dimension-agnostic transition features
    # Movement magnitude (change in absolute values)
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Smoothness (squared difference)
    smoothness = sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # Action penalty (small cost for taking actions)
    action_cost = 0.01 * action

    # Stage weights based on training progress
    # Early: focus on movement and exploration
    # Late: focus on smoothness and stability
    movement_weight = 1.0 - 0.5 * training_progress
    smoothness_weight = 0.5 * training_progress
    action_weight = 0.1 * (1.0 - 0.5 * training_progress)

    # Combine components
    reward = (
        movement_weight * movement
        - smoothness_weight * smoothness
        - action_weight * action_cost
    )

    return float(reward)