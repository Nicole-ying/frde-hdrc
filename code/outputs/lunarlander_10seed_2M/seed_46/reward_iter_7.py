def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Change in absolute values (encourages moving towards zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change magnitude (penalizes large changes)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small cost for taking actions)
    action_cost = 0.01 * float(action)
    
    # Feature 4: Contact bonus (from info if available, otherwise from obs)
    # Using last two obs dimensions as contact indicators (0 or 1)
    contact_bonus = 0.0
    if len(o) >= 2:
        prev_contact = o[-2] + o[-1]
        curr_contact = n[-2] + n[-1]
        contact_bonus = curr_contact - prev_contact  # positive if more contact
    
    # Stage-based weights that evolve with training_progress
    # Stage 1 (early): Focus on reducing absolute values and exploration
    # Stage 2 (mid): Balance between stability and progress
    # Stage 3 (late): Fine-tune for precision and contact
    
    if training_progress < 0.3:
        # Early stage: encourage movement towards zero, small action penalty
        w_abs = 1.0
        w_sq = -0.5
        w_action = -0.1
        w_contact = 0.2
    elif training_progress < 0.7:
        # Mid stage: balance exploration and exploitation
        w_abs = 0.8
        w_sq = -0.3
        w_action = -0.05
        w_contact = 0.5
    else:
        # Late stage: fine-tune for stability and contact
        w_abs = 0.5
        w_sq = -0.2
        w_action = -0.02
        w_contact = 1.0
    
    # Combine components
    reward = (w_abs * abs_change +
              w_sq * sq_change +
              w_action * action_cost +
              w_contact * contact_bonus)
    
    return reward