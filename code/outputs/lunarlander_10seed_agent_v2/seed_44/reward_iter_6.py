def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement towards target: sum of absolute changes
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 2. Smoothness: squared change penalty
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 3. Action penalty (encourage efficient actions)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.2
    
    # Stage-based weights
    if training_progress < 0.3:
        # Early stage: focus on movement and exploration
        w_abs = 1.0
        w_smooth = -0.1
        w_action = -0.2
    elif training_progress < 0.7:
        # Middle stage: balance movement and smoothness
        w_abs = 0.8
        w_smooth = -0.3
        w_action = -0.3
    else:
        # Late stage: fine-tune with smoothness and efficiency
        w_abs = 0.5
        w_smooth = -0.5
        w_action = -0.4
    
    # Compute reward components
    reward = w_abs * abs_diff + w_smooth * squared_change + w_action * action_cost
    
    # Add small exploration bonus in early stages
    if training_progress < 0.3:
        reward += 0.1 * sum(abs(n[i]) for i in range(len(n)))
    
    return float(reward)