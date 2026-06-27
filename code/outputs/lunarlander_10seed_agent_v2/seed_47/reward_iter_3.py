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
    action_cost = -0.01 if action != 0 else 0.0
    
    # Feature 5: Survival bonus - reward for staying alive
    survival_bonus = 0.1
    
    # Feature 6: Stability signal - penalize extreme values (prevents tumbling)
    stability = -sum(abs(n[i]) for i in range(len(o))) * 0.05
    
    # Stage-based weighting
    if training_progress < 0.3:
        # Early exploration phase
        w_movement = 1.0
        w_abs_change = 0.5
        w_smoothness = 0.1
        w_action = -0.01
        w_survival = 0.5
        w_stability = 0.1
    elif training_progress < 0.7:
        # Middle refinement phase
        w_movement = 0.5
        w_abs_change = 1.0
        w_smoothness = 0.3
        w_action = -0.02
        w_survival = 0.8
        w_stability = 0.3
    else:
        # Late precision phase
        w_movement = 0.2
        w_abs_change = 0.5
        w_smoothness = 1.0
        w_action = -0.03
        w_survival = 1.0
        w_stability = 0.5
    
    # Combine components
    reward = (w_movement * movement +
              w_abs_change * abs_change +
              w_smoothness * smoothness +
              w_action * action_cost +
              w_survival * survival_bonus +
              w_stability * stability)
    
    return float(reward)