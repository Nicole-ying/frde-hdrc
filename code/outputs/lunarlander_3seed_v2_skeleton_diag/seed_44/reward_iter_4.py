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
    # This is a constant positive reward per step that increases with training progress
    survival_bonus = 0.1 * (1.0 + training_progress)
    
    # Feature 6: Progress signal - reward reduction in observation magnitude
    # This is a denser version of abs_change that works better
    progress_signal = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 7: Angular stability signal - penalize large angular changes
    # Use the last two dimensions as proxies for angular velocity and angle
    # For 8-dim obs, indices 4 and 5 are angle and angular velocity
    angular_penalty = 0.0
    if len(o) >= 6:
        angular_penalty = abs(n[4]) + abs(n[5])  # penalize non-zero angle and angular velocity
    
    # Feature 8: Ground contact bonus - reward stable landing
    # For 8-dim obs, indices 6 and 7 are leg contact indicators
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = n[6] + n[7]  # sum of leg contacts (0, 1, or 2)
    
    # Stage weights based on training progress
    # Early exploration: encourage movement and exploration
    # Late exploitation: encourage stability and precision
    w1 = 1.0 - 0.5 * training_progress  # weight for abs_change (decreases over time)
    w2 = 0.1 + 0.5 * training_progress  # weight for stability (increases over time)
    w3 = 0.05  # constant action penalty
    w4 = 0.2 * training_progress  # weight for velocity penalty (increases over time)
    w5 = 1.0  # survival bonus weight
    w6 = 0.5 * (1.0 - training_progress)  # weight for progress signal (decreases over time)
    w7 = 0.5 + 1.0 * training_progress  # weight for angular penalty (increases over time)
    w8 = 0.5 + 1.5 * training_progress  # weight for contact bonus (increases over time)
    
    # Combine components - added angular stability and contact bonus
    reward = (w1 * abs_change 
              - w2 * sq_change 
              - w3 * action_cost 
              - w4 * vel_change 
              + w5 * survival_bonus
              + w6 * progress_signal
              - w7 * angular_penalty
              + w8 * contact_bonus)
    
    return reward