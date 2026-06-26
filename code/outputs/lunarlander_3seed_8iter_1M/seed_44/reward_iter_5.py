def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Dimension-agnostic transition features
    # Feature 1: Movement towards target (reducing absolute values) - amplified
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Feature 2: Smoothness (penalize large changes) - reduced to avoid over-penalizing
    smoothness = -0.5 * sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # Feature 3: Action cost (penalize engine usage) - reduced
    action_cost = -0.005 * float(action)

    # Feature 4: Distance progress (reward being closer to origin) - added back from iteration 2 which had best structure
    current_dist = sum(abs(o[i]) for i in range(len(o)))
    next_dist = sum(abs(n[i]) for i in range(len(o)))
    dist_progress = current_dist - next_dist

    # Stage weights based on training_progress
    # Early: focus on movement and exploration
    # Late: focus on smoothness and precision
    w_movement = 2.0 - 1.0 * training_progress  # Increased movement weight
    w_smoothness = 0.3 * training_progress  # Reduced smoothness weight
    w_action = 0.01
    w_dist = 0.5  # Consistent weight for distance progress

    # Combine components
    reward = w_movement * movement + w_smoothness * smoothness + w_action * action_cost + w_dist * dist_progress

    return reward