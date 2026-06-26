def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Dimension-agnostic transition features

    # Feature 1: Movement towards target (reducing absolute values)
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Feature 2: Smoothness (penalize large changes)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # Feature 3: Action cost (penalize engine usage)
    action_cost = -0.01 * float(action)

    # Feature 4: Progress towards zero - reward being closer to origin in state space
    # This is a more direct measure of task progress
    current_dist = sum(abs(o[i]) for i in range(len(o)))
    next_dist = sum(abs(n[i]) for i in range(len(o)))
    dist_progress = current_dist - next_dist

    # Stage weights based on training_progress
    # Early: focus on movement and exploration
    # Late: focus on smoothness and precision
    w_movement = 1.0 - 0.5 * training_progress
    w_smoothness = 0.5 * training_progress
    w_action = 0.01
    w_dist = 0.5  # Consistent weight for distance progress

    # Combine components (removed vel_change which hurt performance)
    reward = (w_movement * movement + 
              w_smoothness * smoothness + 
              w_action * action_cost + 
              w_dist * dist_progress)

    return reward