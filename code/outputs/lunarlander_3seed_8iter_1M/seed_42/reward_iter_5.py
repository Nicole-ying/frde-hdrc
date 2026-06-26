def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays from observations
    o = obs
    n = next_obs
    
    # Dimension-agnostic transition features
    
    # Feature 1: Change in absolute values (penalize growing magnitudes, reward shrinking)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness of state changes (penalize large jumps)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (discourage excessive action usage)
    action_cost = -0.01 * float(action)
    
    # Feature 4: Progress-based exploration bonus (encourage movement early, stability later)
    exploration_bonus = sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Stage weights based on training progress
    # Early training: focus on exploration and reducing magnitudes
    # Late training: focus on smoothness and stability
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Combine components with stage-adaptive weights
    reward = (
        early_weight * (0.5 * abs_change + 0.3 * exploration_bonus) +
        late_weight * (0.7 * smoothness + 0.3 * action_cost) +
        0.1 * (1.0 - training_progress) * exploration_bonus  # persistent small exploration
    )
    
    return reward