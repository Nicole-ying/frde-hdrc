def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages movement toward desirable states)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large sudden changes, encourages smooth transitions)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (discourages excessive action usage)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.2
    
    # Feature 4: Ground contact bonus (from info or observation)
    # Last two elements of obs are ground contact indicators
    ground_contact = o[-1] + o[-2] if len(o) >= 2 else 0.0
    
    # Stage-based weights
    # Early stage: focus on exploration and smooth movement
    # Middle stage: balance movement and control
    # Late stage: emphasize precision and ground contact
    
    if training_progress < 0.3:
        # Early training: encourage movement and exploration
        w_abs = 1.0
        w_sq = -0.1
        w_action = -0.2
        w_contact = 0.5
    elif training_progress < 0.7:
        # Middle training: balance
        w_abs = 0.8
        w_sq = -0.3
        w_action = -0.4
        w_contact = 1.0
    else:
        # Late training: precision and landing
        w_abs = 0.5
        w_sq = -0.5
        w_action = -0.6
        w_contact = 2.0
    
    # Combine components
    reward = (w_abs * abs_diff + 
              w_sq * sq_change + 
              w_action * action_cost + 
              w_contact * ground_contact)
    
    return reward