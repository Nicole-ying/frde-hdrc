def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement magnitude (sum of squared differences)
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 2: Absolute value change (sum of absolute differences)
    abs_change = sum(abs(n[i]) - abs(o[i]) for i in range(len(o)))
    
    # Feature 3: Action cost (small penalty for taking actions)
    action_cost = 0.01 * action
    
    # Stage-based weights
    # Early stage: encourage exploration (movement)
    # Middle stage: balance movement and absolute change
    # Late stage: focus on stability (small absolute changes)
    if training_progress < 0.3:
        # Early training - encourage movement and exploration
        w_movement = 1.0
        w_abs_change = 0.1
        w_action = -0.01
    elif training_progress < 0.7:
        # Middle training - balance exploration and exploitation
        w_movement = 0.5
        w_abs_change = 0.5
        w_action = -0.02
    else:
        # Late training - focus on stability and precision
        w_movement = 0.1
        w_abs_change = 1.0
        w_action = -0.03
    
    # Combine components
    reward = (w_movement * movement + 
              w_abs_change * abs_change + 
              w_action * action_cost)
    
    return reward