def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement toward stability - sum of absolute changes
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness - squared changes penalize large jumps
    sq_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost - penalize frequent actions
    action_cost = 0.0
    if action in [1, 2, 3]:
        action_cost = 0.1
    
    # Feature 4: Contact bonus - from info if available, else from obs
    contact_bonus = 0.0
    if len(o) >= 8:
        # Last two obs elements might indicate contact
        contact_bonus = (o[6] + o[7]) * 0.5
    
    # Stage weights based on training progress
    # Early stage: focus on exploration and movement
    # Middle stage: balance stability and action efficiency
    # Late stage: emphasize precision and contact
    
    if training_progress < 0.3:
        # Early exploration
        w1 = 1.0  # Movement reward weight
        w2 = -0.5  # Smoothness penalty weight
        w3 = -0.2  # Action cost weight
        w4 = 0.3   # Contact bonus weight
    elif training_progress < 0.7:
        # Middle refinement
        w1 = 0.7
        w2 = -0.8
        w3 = -0.4
        w4 = 0.6
    else:
        # Late precision
        w1 = 0.3
        w2 = -1.0
        w3 = -0.6
        w4 = 1.0
    
    # Combine components
    reward = (w1 * abs_diff + 
              w2 * sq_diff + 
              w3 * action_cost + 
              w4 * contact_bonus)
    
    return reward