def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Create generic arrays for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability (reducing absolute values)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff / len(o)
    
    # Component 2: Smoothness of transition (small changes are good)
    delta = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_reward = -delta / len(o)
    
    # Component 3: Action cost penalty (encourage efficient actions)
    action_cost = -0.01 * float(action)
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on movement and exploration
    # Late training: focus on smoothness and precision
    stage1_weight = 1.0 - training_progress  # Decreases over time
    stage2_weight = training_progress  # Increases over time
    
    # Combine components with stage weights
    reward = (
        stage1_weight * movement_reward +
        stage2_weight * smoothness_reward +
        action_cost
    )
    
    return reward