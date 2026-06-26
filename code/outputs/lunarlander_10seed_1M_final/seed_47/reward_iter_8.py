def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Generic arrays for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability - penalize large changes in absolute values
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness - penalize large squared differences
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Component 3: Action cost - penalize taking actions
    action_cost = 0.0
    if action in [1, 2, 3]:  # Any non-zero action
        action_cost = 0.1
    
    # Stage-based weights that evolve with training
    stage1_weight = 1.0 - training_progress  # Early: focus on stability
    stage2_weight = training_progress  # Later: focus on smoothness
    
    # Combine components with stage weights
    reward = (
        stage1_weight * abs_change * 0.5 +
        stage2_weight * (-squared_diff * 0.1) +
        -action_cost * 0.05
    )
    
    return reward