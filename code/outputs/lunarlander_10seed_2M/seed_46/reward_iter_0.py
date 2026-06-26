def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability - penalize large changes
    # Use sum of squared differences as a generic transition measure
    delta = n - o
    movement_cost = sum(d * d for d in delta)
    
    # Component 2: Encourage convergence - reward when absolute values decrease
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 3: Small action penalty to encourage efficiency
    action_cost = 0.01 * action
    
    # Component 4: Use ground contact signals from info if available
    # Generic contact bonus
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = 0.5 * info['contact']
    
    # Stage-based weighting
    # Early stage: focus on movement and exploration
    # Late stage: focus on stability and convergence
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Combine components with stage-adaptive weights
    reward = (
        -0.1 * movement_cost * (0.5 + 0.5 * late_weight) +  # Penalize large changes more later
        0.5 * abs_diff * (0.3 + 0.7 * early_weight) +  # Reward convergence more early
        -0.02 * action_cost +  # Small action penalty
        contact_bonus * late_weight  # Contact bonus more important later
    )
    
    return reward