def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude: squared difference between next and current observations
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 2. Absolute value change: sum of absolute differences
    abs_change = sum(abs(n[i]) - abs(o[i]) for i in range(len(o)))
    
    # 3. Action penalty (small cost for taking actions)
    action_cost = 0.01 * action  # action is discrete 0-3
    
    # Stage-based weights
    # Early stage: focus on exploration (movement)
    # Middle stage: balance movement and absolute changes
    # Late stage: focus on stability (small absolute changes)
    
    # Sigmoid-like transition based on training_progress
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage weights
    reward = (
        early_weight * movement * 0.5 +
        mid_weight * abs_change * 0.3 +
        late_weight * (-abs_change) * 0.2 -  # negative abs_change in late stage for stability
        action_cost * 0.1
    )
    
    # Scale reward to reasonable range
    reward = reward * 0.1
    
    return reward