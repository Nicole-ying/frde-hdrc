def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude: sum of squared differences
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 2. Absolute change in state magnitudes
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 3. Action cost (penalize large actions)
    action_cost = 0.0
    if action == 2:  # main engine
        action_cost = 0.1
    elif action in [1, 3]:  # side engines
        action_cost = 0.05
    
    # Stage-based weights
    # Early stage: focus on movement exploration
    # Middle stage: balance movement and state improvement
    # Late stage: focus on convergence (small changes)
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Component rewards
    movement_reward = movement * 0.5
    abs_change_reward = abs_change * 0.3
    action_penalty = -action_cost
    
    # Combine with stage weights
    reward = (
        early_weight * (movement_reward + abs_change_reward) +
        mid_weight * (abs_change_reward - movement_reward * 0.2) +
        late_weight * (-abs_change * 0.1 - movement * 0.3) +
        action_penalty
    )
    
    return reward