def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays as generic o and n
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude: sum of squared differences
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 2. Absolute value change: encourage moving toward zero if beneficial
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 3. Action penalty: small cost for taking actions
    action_cost = 0.01 * abs(action - 1.5)  # center around 1.5 for discrete 0-3
    
    # Stage-based weighting
    # Early exploration: focus on movement and abs_change
    # Late exploitation: reduce action cost importance
    early_weight = max(0.0, 1.0 - training_progress)
    late_weight = min(1.0, training_progress)
    
    # Combine components with stage weights
    reward = (
        early_weight * (0.5 * movement + 0.3 * abs_change) +
        late_weight * (0.2 * movement + 0.1 * abs_change - 0.5 * action_cost)
    )
    
    return reward