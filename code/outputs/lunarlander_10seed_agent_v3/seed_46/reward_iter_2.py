def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features - dimension agnostic
    # Feature 1: Change in absolute values (encourages moving towards zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change magnitude (encourages smooth transitions)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (discourages excessive action usage)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0
    elif action == 1 or action == 3:
        action_cost = 0.1
    elif action == 2:
        action_cost = 0.2
    
    # Feature 4: Ground contact bonus from info
    ground_contact_bonus = 0.0
    if 'ground_contact' in info:
        ground_contact_bonus = info['ground_contact'] * 0.5
    
    # Stage-based weights that evolve with training_progress
    # Stage 1 (early): Focus on reducing absolute values and exploring
    # Stage 2 (mid): Balance between stability and action efficiency
    # Stage 3 (late): Fine-tune with smooth transitions
    
    # Compute stage weights
    if training_progress < 0.3:
        # Early stage: encourage exploration and reducing absolute values
        w_abs = 1.0
        w_sq = -0.1
        w_action = -0.05
        w_contact = 0.2
    elif training_progress < 0.7:
        # Middle stage: balance stability and efficiency
        w_abs = 0.7
        w_sq = -0.3
        w_action = -0.1
        w_contact = 0.4
    else:
        # Late stage: fine-tune for smooth, efficient control
        w_abs = 0.3
        w_sq = -0.5
        w_action = -0.15
        w_contact = 0.6
    
    # Combine components
    reward = (w_abs * abs_change + 
              w_sq * sq_change + 
              w_action * action_cost + 
              w_contact * ground_contact_bonus)
    
    return reward