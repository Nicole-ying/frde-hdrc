def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability (reducing absolute values)
    # Positive when absolute values decrease, negative when they increase
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_change / len(o)
    
    # Component 2: Smoothness - penalize large changes (encourages gentle control)
    diff_sq = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -diff_sq / (len(o) * 10.0)
    
    # Component 3: Action cost - penalize engine usage
    # Actions: 0=do nothing, 1=left, 2=main, 3=right
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = -0.5
    elif action in [1, 3]:  # Side engines
        action_cost = -0.2
    
    # Component 4: Ground contact bonus (from info or obs)
    # Obs indices 6 and 7 are ground contact indicators
    ground_contact = 0.0
    if len(o) >= 8:
        # Check if both legs are on ground (indices 6 and 7)
        if n[6] > 0.5 and n[7] > 0.5:
            ground_contact = 1.0
        elif n[6] > 0.5 or n[7] > 0.5:
            ground_contact = 0.3
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on movement and smoothness
    # Middle stage: balance movement with action efficiency
    # Late stage: emphasize ground contact and stability
    
    if training_progress < 0.3:
        # Early exploration: reward movement toward stability
        w_movement = 2.0
        w_smoothness = 0.5
        w_action = 0.3
        w_ground = 0.1
    elif training_progress < 0.7:
        # Middle stage: refine control
        w_movement = 1.5
        w_smoothness = 1.0
        w_action = 0.5
        w_ground = 0.5
    else:
        # Late stage: focus on landing success
        w_movement = 1.0
        w_smoothness = 1.5
        w_action = 0.3
        w_ground = 2.0
    
    # Combine components
    reward = (w_movement * movement_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_ground * ground_contact)
    
    return reward