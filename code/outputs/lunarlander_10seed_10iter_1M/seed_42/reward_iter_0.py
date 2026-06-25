def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement toward stability - reduction in absolute values
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness - small squared changes
    sq_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalize large actions)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # Feature 4: Terminal signal from info (if available)
    terminal_signal = 0.0
    if info:
        if 'terminal' in info:
            terminal_signal = 1.0 if info['terminal'] else 0.0
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on exploration and reducing absolute values
    # Middle stage: balance smoothness and stability
    # Late stage: refine with action efficiency
    
    # Sigmoid-like transition based on training_progress
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Component rewards
    stability_reward = abs_diff * 0.5  # Positive when moving toward zero
    smoothness_reward = -sq_diff * 0.1  # Negative for large changes
    action_penalty = -action_cost * 0.2
    terminal_bonus = terminal_signal * 2.0
    
    # Combine with stage weights
    reward = (
        early_weight * stability_reward +
        mid_weight * (stability_reward * 0.3 + smoothness_reward * 0.7) +
        late_weight * (smoothness_reward * 0.4 + action_penalty * 0.6) +
        terminal_bonus * 0.1
    )
    
    # Normalize by stage weights sum to keep scale consistent
    weight_sum = early_weight + mid_weight + late_weight + 0.1
    if weight_sum > 0:
        reward = reward / weight_sum
    
    return reward