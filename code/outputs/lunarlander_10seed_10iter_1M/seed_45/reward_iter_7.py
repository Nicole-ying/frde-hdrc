def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages reaching states with smaller magnitudes)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (captures overall movement magnitude)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalize large actions)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0
    elif action == 1 or action == 3:
        action_cost = 0.1
    elif action == 2:
        action_cost = 0.2
    
    # Feature 4: Contact signals from info or last two obs dimensions
    contact_signal = 0.0
    if len(o) >= 8:
        contact_signal = n[6] + n[7]  # Sum of ground contact indicators
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on exploration (squared change) and reducing magnitudes
    # Middle stage: balance between all components
    # Late stage: focus on contact signals and stability
    
    if training_progress < 0.3:
        # Early exploration phase
        w_abs = 0.5
        w_sq = 1.0
        w_action = -0.3
        w_contact = 0.1
    elif training_progress < 0.7:
        # Middle phase
        w_abs = 0.3
        w_sq = 0.5
        w_action = -0.5
        w_contact = 0.5
    else:
        # Late refinement phase
        w_abs = 0.1
        w_sq = 0.2
        w_action = -0.8
        w_contact = 1.0
    
    # Combine components
    reward = (w_abs * abs_change + 
              w_sq * squared_change + 
              w_action * action_cost + 
              w_contact * contact_signal)
    
    return reward