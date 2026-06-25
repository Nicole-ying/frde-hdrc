def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement towards zero (absolute value reduction)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness of transition (small changes are better)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalize large actions)
    action_cost = -0.01 * float(action)
    
    # Feature 4: State magnitude penalty (discourage extreme states)
    state_magnitude = -sum(n[i] ** 2 for i in range(len(o))) * 0.001
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on exploration and smooth transitions
    # Middle stage: balance movement and stability
    # Late stage: emphasize precision and state quality
    
    # Sigmoid-like transition functions
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    middle_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage-appropriate weights
    reward = (
        early_weight * (0.5 * abs_diff + 0.3 * smoothness + 0.2 * action_cost) +
        middle_weight * (0.4 * abs_diff + 0.3 * smoothness + 0.2 * action_cost + 0.1 * state_magnitude) +
        late_weight * (0.3 * abs_diff + 0.2 * smoothness + 0.1 * action_cost + 0.4 * state_magnitude)
    )
    
    # Normalize by stage weights sum to keep scale consistent
    weight_sum = early_weight + middle_weight + late_weight
    if weight_sum > 0:
        reward = reward / weight_sum
    
    return float(reward)