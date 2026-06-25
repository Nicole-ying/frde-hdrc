def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large changes in absolute values
    # This encourages smooth transitions and reaching a stable state
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_change / len(o)
    
    # Component 2: Smoothness - penalize large jumps between states
    diff_sq = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -diff_sq / len(o)
    
    # Component 3: Action cost - penalize taking actions (encourages efficiency)
    action_cost = -0.01 * float(action)
    
    # Component 4: Terminal state bonus from info if available
    terminal_bonus = 0.0
    if info:
        # Check for common termination signals
        if 'is_success' in info and info['is_success']:
            terminal_bonus = 10.0
        elif 'TimeLimit.truncated' in info and info['TimeLimit.truncated']:
            terminal_bonus = -5.0
    
    # Stage weights based on training_progress (0.0 to 1.0)
    # Early training: focus on exploration and movement
    # Mid training: balance smoothness and stability
    # Late training: emphasize precision and terminal outcomes
    
    if training_progress < 0.3:
        # Early stage: encourage exploration and movement
        w_movement = 1.0
        w_smoothness = 0.1
        w_action = 0.05
        w_terminal = 0.0
    elif training_progress < 0.7:
        # Middle stage: balance all components
        w_movement = 0.6
        w_smoothness = 0.3
        w_action = 0.1
        w_terminal = 0.2
    else:
        # Late stage: focus on precision and terminal outcomes
        w_movement = 0.3
        w_smoothness = 0.4
        w_action = 0.1
        w_terminal = 0.5
    
    # Combine components with stage weights
    reward = (w_movement * movement_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_terminal * terminal_bonus)
    
    return float(reward)