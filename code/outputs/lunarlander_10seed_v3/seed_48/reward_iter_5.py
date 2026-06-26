def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Change in absolute values (encourages moving towards zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change magnitude (penalizes large changes)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalizes action usage)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0  # no action
    elif action == 1 or action == 3:
        action_cost = 0.5  # side engine
    elif action == 2:
        action_cost = 1.0  # main engine
    
    # Feature 4: Contact signals from info (if available)
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = info['contact'] * 0.5
    
    # Stage-based weights that evolve with training_progress
    # Stage 1 (early): Focus on reducing absolute values and exploring
    w1_early = 1.0 - 0.5 * training_progress
    w2_early = 0.2 * (1.0 - training_progress)
    w3_early = 0.1 * (1.0 - training_progress)
    
    # Stage 2 (late): Focus on stability and contact
    w1_late = 0.5 * training_progress
    w2_late = 0.3 * training_progress
    w3_late = 0.2 * training_progress
    
    # Combine stage weights
    w1 = w1_early + w1_late
    w2 = w2_early + w2_late
    w3 = w3_early + w3_late
    
    # Compute reward components
    reward_abs = abs_change * w1
    reward_sq = -sq_change * w2  # negative because we want to minimize change
    reward_action = -action_cost * w3
    reward_contact = contact_bonus * (0.5 + 0.5 * training_progress)
    
    # Combine all components
    total_reward = reward_abs + reward_sq + reward_action + reward_contact
    
    return total_reward