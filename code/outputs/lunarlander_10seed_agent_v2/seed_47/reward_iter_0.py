def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement magnitude - sum of absolute differences
    movement = sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Feature 2: Change in absolute values (encourages moving toward zero-centered states)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 3: Smoothness penalty - large jumps are discouraged
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 4: Action penalty (small action cost for discrete actions)
    # For discrete(4), penalize non-zero actions slightly
    action_cost = -0.01 if action != 0 else 0.0
    
    # Stage-based weighting
    # Early stage: focus on exploration and movement
    # Middle stage: balance movement with stability
    # Late stage: prioritize precision and smoothness
    
    if training_progress < 0.3:
        # Early exploration phase
        w_movement = 1.0
        w_abs_change = 0.5
        w_smoothness = -0.1
        w_action = -0.01
    elif training_progress < 0.7:
        # Middle refinement phase
        w_movement = 0.5
        w_abs_change = 1.0
        w_smoothness = 0.3
        w_action = -0.02
    else:
        # Late precision phase
        w_movement = 0.2
        w_abs_change = 0.5
        w_smoothness = 1.0
        w_action = -0.03
    
    # Combine components
    reward = (w_movement * movement +
              w_abs_change * abs_change +
              w_smoothness * smoothness +
              w_action * action_cost)
    
    # Add small constant to encourage exploration
    reward = reward + 0.1
    
    return float(reward)