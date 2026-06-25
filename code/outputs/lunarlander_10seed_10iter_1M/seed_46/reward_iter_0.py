def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability - penalize large changes in absolute values
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.1
    
    # Component 2: Smoothness - penalize large squared changes
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_change * 0.05
    
    # Component 3: Action cost - small penalty for taking actions
    action_cost = -0.01 * (1 if action != 0 else 0)
    
    # Component 4: Contact bonus - reward for ground contact (from info or obs)
    # Using last two elements of obs which appear to be ground contact indicators
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = (o[6] + o[7]) * 0.5
    
    # Stage weights based on training_progress
    stage1_weight = max(0.0, 1.0 - training_progress * 2.0)  # Early: focus on movement
    stage2_weight = min(1.0, training_progress * 2.0) * (1.0 - max(0.0, training_progress * 2.0 - 1.0))  # Middle: balance
    stage3_weight = max(0.0, training_progress * 2.0 - 1.0)  # Late: focus on contact
    
    # Combine components with stage weights
    reward = (
        stage1_weight * movement_reward +
        stage2_weight * (smoothness_penalty + action_cost) +
        stage3_weight * contact_bonus
    )
    
    return reward