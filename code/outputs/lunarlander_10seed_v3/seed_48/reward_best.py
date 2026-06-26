def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability - penalize large changes in absolute values
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness - penalize large squared differences
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Component 3: Action penalty - discourage excessive action usage
    action_penalty = 0.0
    if action == 2:  # main engine
        action_penalty = -0.1
    elif action in [1, 3]:  # side engines
        action_penalty = -0.05
    
    # Component 4: Ground contact bonus (from info or obs)
    ground_contact_bonus = 0.0
    if len(o) >= 8:
        leg1_contact = o[6] if len(o) > 6 else 0.0
        leg2_contact = o[7] if len(o) > 7 else 0.0
        ground_contact_bonus = (leg1_contact + leg2_contact) * 0.5
    
    # Stage weights based on training_progress
    # Early stage: focus on stability and smoothness
    # Late stage: focus on ground contact and action efficiency
    early_weight = max(0.0, 1.0 - training_progress * 2.0)
    late_weight = min(1.0, training_progress * 2.0)
    
    # Combine components with stage-adaptive weights
    reward = (
        early_weight * (0.5 * abs_change - 0.3 * squared_diff) +
        late_weight * (ground_contact_bonus + action_penalty) +
        action_penalty * 0.1  # small constant action penalty
    )
    
    return reward