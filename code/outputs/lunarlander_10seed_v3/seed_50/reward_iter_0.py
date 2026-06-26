def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages movement toward center)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large jumps, encourages smooth transitions)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (encourages efficient actions)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.2
    
    # Feature 4: Ground contact bonus (from info or observation)
    # Check if legs are in contact (indices 6 and 7 in observation)
    leg_contact = o[6] + o[7]  # Sum of leg contact indicators
    
    # Stage-based weights
    # Early stage: focus on movement and exploration
    w1 = 1.0 - 0.5 * training_progress  # abs_change weight decreases over time
    w2 = 0.1 + 0.3 * training_progress   # squared_change weight increases
    w3 = 0.2 * (1.0 - training_progress) # action cost decreases
    w4 = 0.5 * training_progress          # leg contact bonus increases
    
    # Combine components
    reward = (w1 * abs_change 
              - w2 * squared_change 
              - w3 * action_cost 
              + w4 * leg_contact)
    
    return reward