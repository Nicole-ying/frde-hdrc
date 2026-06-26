def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement towards center - sum of absolute changes
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))
    
    # Feature 2: Smoothness - squared difference between consecutive observations
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small cost for taking actions)
    action_cost = -0.01 * abs(action)
    
    # Feature 4: Ground contact bonus (from info or observation)
    ground_contact = 0.0
    if len(o) >= 8:
        ground_contact = o[6] + o[7]  # leg contact indicators
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on exploration and movement
    # Middle stage: balance movement and stability
    # Late stage: prioritize precision and ground contact
    
    w1 = 0.5 + 0.3 * training_progress  # Movement weight increases
    w2 = 0.3 - 0.2 * training_progress  # Smoothness weight decreases
    w3 = 0.1 * (1.0 - training_progress)  # Action cost decreases
    w4 = 0.1 + 0.4 * training_progress  # Ground contact weight increases
    
    # Combine components
    reward = (w1 * abs_change + 
              w2 * smoothness + 
              w3 * action_cost + 
              w4 * ground_contact)
    
    return reward