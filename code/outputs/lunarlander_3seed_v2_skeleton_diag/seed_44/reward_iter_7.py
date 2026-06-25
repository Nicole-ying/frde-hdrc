def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Dimension-agnostic transition features
    
    # Feature 1: Movement toward zero (encourages stabilization)
    # Positive when absolute values decrease
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness penalty (discourages large jerky changes)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalizes engine usage)
    # For discrete actions, action is a scalar integer
    action_cost = 0.01 * float(action)
    
    # Feature 4: Velocity change signal (encourages slowing down)
    # Use the difference in consecutive observations as a proxy for velocity
    vel_change = sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Feature 5: Survival bonus (encourages staying alive longer)
    survival_bonus = 0.1 * (1.0 + training_progress)
    
    # Feature 6: Progress signal - reward reduction in observation magnitude
    progress_signal = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 7: Angular stability signal - penalize large angular changes
    # Use the last two dimensions as proxies for angular velocity and angle
    angular_penalty = 0.0
    if len(o) >= 6:
        angular_penalty = abs(n[4]) + abs(n[5])
    
    # Feature 8: Ground contact bonus - reward stable landing
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = n[6] + n[7]
    
    # Feature 9: Height penalty - penalize being high up (encourage descent)
    height_penalty = 0.0
    if len(o) >= 2:
        height_penalty = abs(n[1])
    
    # Stage weights based on training progress
    # Historical analysis:
    # - Iteration 4 (score=-24.5) had moderate weights for all components
    # - Iteration 5 (score=-111.2) added height_penalty with large weight, causing early termination
    # - Iteration 6 (score=-13.722) removed height_penalty, improved score
    # - The best score (-13.722) came from iteration 6 which had moderate weights
    # - Key insight: height_penalty is harmful, contact_bonus needs moderate weight
    # - abs_change and progress_signal are redundant - merge them with higher weight
    # - vel_change penalty may be too aggressive - reduce it
    
    w1 = 1.5 - 0.5 * training_progress  # weight for abs_change (increased, decreases over time)
    w2 = 0.1 + 0.5 * training_progress  # weight for stability (increases over time)
    w3 = 0.05  # constant action penalty
    w4 = 0.1 * training_progress  # weight for velocity penalty (reduced, increases over time)
    w5 = 1.0  # survival bonus weight
    w6 = 0.0  # progress signal weight - REMOVED (redundant with abs_change)
    w7 = 0.3 + 0.5 * training_progress  # weight for angular penalty (reduced, increases over time)
    w8 = 0.3 + 1.0 * training_progress  # weight for contact bonus (reduced, increases over time)
    w9 = 0.0  # height penalty weight - REMOVED (caused early termination)
    
    # Combine components - tuned from iteration 6 structure
    reward = (w1 * abs_change 
              - w2 * sq_change 
              - w3 * action_cost 
              - w4 * vel_change 
              + w5 * survival_bonus
              + w6 * progress_signal
              - w7 * angular_penalty
              + w8 * contact_bonus
              - w9 * height_penalty)
    
    return reward