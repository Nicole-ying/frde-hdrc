def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large changes in absolute values
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.1
    
    # Component 2: Smoothness - penalize large squared changes
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_change * 0.5
    
    # Component 3: Action cost - penalize taking actions (encourage efficiency)
    action_cost = -0.01 * float(action)
    
    # Component 4: Ground contact bonus from info
    ground_contact = 0.0
    if 'ground_contact' in info:
        ground_contact = float(info['ground_contact']) * 2.0
    
    # Stage weights based on training progress
    stage1_weight = max(0.0, 1.0 - training_progress * 2.0)  # Early: focus on movement
    stage2_weight = 1.0 - abs(training_progress - 0.5) * 2.0  # Middle: balance
    stage3_weight = max(0.0, training_progress * 2.0 - 1.0)  # Late: focus on stability
    
    # Combine components with stage weights
    reward = (
        stage1_weight * movement_reward +
        stage2_weight * (smoothness_penalty + action_cost) +
        stage3_weight * ground_contact
    )
    
    return reward