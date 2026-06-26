def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement magnitude - sum of absolute differences
    abs_diff = sum(abs(o[i] - n[i]) for i in range(len(o)))
    
    # Feature 2: Squared displacement - penalizes large changes
    sq_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Directional consistency - positive change in same direction
    direction_change = sum((n[i] - o[i]) * o[i] for i in range(len(o)))
    
    # Feature 4: Action cost - penalize frequent actions
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0
    elif action == 1 or action == 3:
        action_cost = 0.1
    elif action == 2:
        action_cost = 0.2
    
    # Feature 5: Stability measure - penalize oscillating behavior
    stability = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Stage-based weighting
    # Early stage: focus on exploration and movement
    # Middle stage: balance movement and stability
    # Late stage: prioritize stability and precision
    if training_progress < 0.3:
        # Early exploration
        w_abs_diff = 0.3
        w_sq_diff = -0.1
        w_direction = 0.4
        w_action = -0.2
        w_stability = 0.2
    elif training_progress < 0.7:
        # Middle stage - balanced
        w_abs_diff = 0.2
        w_sq_diff = -0.2
        w_direction = 0.3
        w_action = -0.3
        w_stability = 0.4
    else:
        # Late stage - precision
        w_abs_diff = 0.1
        w_sq_diff = -0.3
        w_direction = 0.2
        w_action = -0.4
        w_stability = 0.6
    
    # Compute reward components
    reward = 0.0
    reward += w_abs_diff * abs_diff
    reward += w_sq_diff * sq_diff
    reward += w_direction * direction_change
    reward += w_action * action_cost
    reward += w_stability * stability
    
    # Normalize reward to reasonable range
    # Scale factor based on observation dimension
    scale = 1.0 / (1.0 + len(o) * 0.1)
    reward = reward * scale
    
    return reward