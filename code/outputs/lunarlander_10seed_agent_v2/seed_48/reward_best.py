def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability (reducing absolute values)
    # Positive when absolute values decrease
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness of transition (small changes are good)
    # Negative when changes are large
    change_penalty = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Component 3: Action cost (prefer smaller action indices)
    # Actions are 0,1,2,3 - penalize higher indices slightly
    action_cost = -0.01 * float(action)
    
    # Component 4: Ground contact bonus from info
    # Check if legs have ground contact (indices 6 and 7 in obs)
    leg_contact_bonus = 0.0
    if len(o) >= 8:
        leg1_contact = o[6]
        leg2_contact = o[7]
        leg_contact_bonus = 0.5 * (leg1_contact + leg2_contact)
    
    # Stage weights based on training_progress
    # Early stage: focus on reducing absolute values and smoothness
    # Middle stage: balance all components
    # Late stage: emphasize ground contact and stability
    
    if training_progress < 0.3:
        # Early exploration
        w1 = 1.0  # abs_diff
        w2 = 0.5  # change_penalty
        w3 = 0.1  # action_cost
        w4 = 0.2  # leg_contact
    elif training_progress < 0.7:
        # Middle refinement
        w1 = 0.8
        w2 = 0.8
        w3 = 0.2
        w4 = 0.5
    else:
        # Late optimization
        w1 = 0.5
        w2 = 0.3
        w3 = 0.1
        w4 = 1.0
    
    # Combine components
    reward = (w1 * abs_diff + 
              w2 * change_penalty + 
              w3 * action_cost + 
              w4 * leg_contact_bonus)
    
    return reward