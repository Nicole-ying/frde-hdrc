def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change magnitude (penalizes large jumps, encourages smooth transitions)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small action cost to encourage efficiency)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.25
    
    # Feature 4: Contact bonus (from info or observation)
    # Last two obs dimensions are ground contact indicators
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = n[6] + n[7]  # Sum of leg contact indicators
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance movement with efficiency
    # Late stage: emphasize precision and contact
    
    if training_progress < 0.3:
        # Early exploration phase
        w_abs = 1.0
        w_sq = -0.1
        w_action = -0.2
        w_contact = 0.5
    elif training_progress < 0.6:
        # Middle refinement phase
        w_abs = 0.8
        w_sq = -0.3
        w_action = -0.3
        w_contact = 1.0
    elif training_progress < 0.85:
        # Late precision phase
        w_abs = 0.5
        w_sq = -0.5
        w_action = -0.4
        w_contact = 2.0
    else:
        # Final convergence phase
        w_abs = 0.3
        w_sq = -0.7
        w_action = -0.5
        w_contact = 3.0
    
    # Combine components
    reward = (w_abs * abs_change + 
              w_sq * squared_change + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    return reward