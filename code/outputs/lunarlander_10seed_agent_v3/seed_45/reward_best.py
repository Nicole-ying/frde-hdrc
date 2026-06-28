def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - reward reduction in absolute values
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(min(len(o), len(n))))
    movement_reward = abs_diff * 0.1
    
    # Component 2: Smoothness - penalize large changes
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(min(len(o), len(n))))
    smoothness_penalty = -squared_diff * 0.05
    
    # Component 3: Action cost - penalize taking actions
    action_cost = -0.01 * float(action)
    
    # Component 4: Contact bonus - reward ground contact (indices 6 and 7 in observation)
    contact_bonus = 0.0
    if len(o) >= 8 and len(n) >= 8:
        contact_bonus = (n[6] + n[7]) * 0.5
    
    # Stage weights based on training_progress
    stage1_weight = max(0.0, 1.0 - training_progress * 2.0)
    stage2_weight = 1.0 - abs(training_progress - 0.5) * 2.0
    stage3_weight = max(0.0, training_progress * 2.0 - 1.0)
    
    # Combine components with stage-adaptive weights
    reward = (
        movement_reward * (stage1_weight * 0.6 + stage2_weight * 0.3 + stage3_weight * 0.1) +
        smoothness_penalty * (stage1_weight * 0.2 + stage2_weight * 0.4 + stage3_weight * 0.4) +
        action_cost * (stage1_weight * 0.1 + stage2_weight * 0.1 + stage3_weight * 0.1) +
        contact_bonus * (stage1_weight * 0.1 + stage2_weight * 0.2 + stage3_weight * 0.4)
    )
    
    return reward