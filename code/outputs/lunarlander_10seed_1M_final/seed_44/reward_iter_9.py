def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large changes in absolute values
    # This encourages smooth transitions and avoids erratic behavior
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_quality = abs_change / max(1.0, len(o))
    
    # Component 2: Smoothness - penalize large squared differences
    # This encourages gradual changes and penalizes sudden jumps
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness = -squared_diff / max(1.0, len(o))
    
    # Component 3: Action cost - penalize taking actions (encourages efficiency)
    action_cost = -0.01 * float(action)
    
    # Component 4: Terminal signal from info (if available)
    terminal_bonus = 0.0
    if info and isinstance(info, dict):
        if info.get('terminated', False) or info.get('truncated', False):
            terminal_bonus = 1.0
    
    # Stage-based weights that evolve with training_progress
    # Early training: focus on smoothness and movement quality
    # Late training: focus on terminal outcomes and efficiency
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)  # Decays from 1 to 0
    stage2_weight = min(1.0, 2.0 * training_progress)  # Grows from 0 to 1
    stage3_weight = min(1.0, 4.0 * training_progress) if training_progress > 0.25 else 0.0
    
    # Combine components with stage weights
    reward = (
        stage1_weight * movement_quality * 0.5 +
        stage1_weight * smoothness * 0.3 +
        stage2_weight * action_cost * 0.2 +
        stage3_weight * terminal_bonus * 2.0
    )
    
    return float(reward)