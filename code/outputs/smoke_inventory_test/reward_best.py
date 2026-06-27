def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards goal (based on absolute value reduction)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff / len(o)
    
    # Component 2: Smoothness/stability (penalize large changes)
    delta = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    stability_penalty = -delta / (len(o) * 10.0)
    
    # Component 3: Action cost (small penalty for taking actions)
    action_cost = -0.01 * action
    
    # Stage-based weights
    # Early training: focus on exploration and movement
    # Late training: focus on stability and precision
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Combine components with stage weights
    reward = (
        early_weight * (movement_reward * 0.7 + action_cost * 0.3) +
        late_weight * (stability_penalty * 0.6 + movement_reward * 0.4)
    )
    
    return reward