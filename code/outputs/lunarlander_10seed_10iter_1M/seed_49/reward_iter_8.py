def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude: sum of squared differences
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 2. Absolute value change (signed)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 3. Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Stage-based weights
    # Early exploration: focus on movement and absolute changes
    # Late exploitation: reduce noise, focus on stability
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Combine components with stage-adaptive weights
    reward = (
        early_weight * (0.5 * movement + 0.3 * abs_change) +
        late_weight * (0.2 * movement - 0.1 * abs_change) -
        action_cost
    )
    
    return reward