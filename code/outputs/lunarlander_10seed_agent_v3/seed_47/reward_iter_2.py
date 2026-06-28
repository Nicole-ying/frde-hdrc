def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement towards zero - sum of absolute value reduction
    abs_reduction = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness - squared change magnitude (penalize large jumps)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (encourage efficiency)
    action_cost = 0.0
    if action == 1 or action == 3:
        action_cost = -0.02
    elif action == 2:
        action_cost = -0.05
    
    # Feature 4: Leg contact signal (from info or observation)
    # Last two dimensions of obs are leg contact indicators
    leg_contact = 0.0
    if len(o) >= 8:
        leg_contact_prev = o[6] + o[7]
        leg_contact_now = n[6] + n[7]
        leg_contact = leg_contact_now * 0.5  # Reward for having legs on ground
    
    # Stage weights based on training_progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance movement and smoothness
    # Late stage: prioritize stability and leg contact
    
    if training_progress < 0.3:
        # Early exploration phase
        w_abs = 0.8
        w_smooth = -0.1
        w_action = 0.3
        w_leg = 0.0
    elif training_progress < 0.6:
        # Middle phase - balance
        w_abs = 0.5
        w_smooth = -0.3
        w_action = 0.4
        w_leg = 0.2
    else:
        # Late phase - precision and stability
        w_abs = 0.3
        w_smooth = -0.5
        w_action = 0.5
        w_leg = 0.5
    
    # Combine components
    reward = (w_abs * abs_reduction + 
              w_smooth * squared_change + 
              w_action * action_cost + 
              w_leg * leg_contact)
    
    # Add small exploration bonus early on
    if training_progress < 0.2:
        reward += 0.01
    
    return float(reward)