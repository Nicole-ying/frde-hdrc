def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays for generic processing
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change magnitude (captures overall transition size)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalize large actions)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # Feature 4: Terminal signal from info (if available)
    terminal_bonus = 0.0
    if info and isinstance(info, dict):
        if info.get('terminated', False) or info.get('truncated', False):
            terminal_bonus = 1.0
    
    # Stage weights based on training_progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance movement with precision
    # Late stage: emphasize stability and terminal outcomes
    
    # Linear interpolation between stages
    if training_progress < 0.3:
        # Early stage: encourage exploration and movement
        w_abs = 1.0
        w_sq = 0.5
        w_action = -0.2
        w_terminal = 0.0
    elif training_progress < 0.7:
        # Middle stage: balance exploration with precision
        w_abs = 0.7
        w_sq = 1.0
        w_action = -0.5
        w_terminal = 0.5
    else:
        # Late stage: focus on stability and terminal success
        w_abs = 0.3
        w_sq = 1.5
        w_action = -1.0
        w_terminal = 2.0
    
    # Combine components
    reward = (w_abs * abs_change +
              w_sq * sq_change +
              w_action * action_cost +
              w_terminal * terminal_bonus)
    
    return reward