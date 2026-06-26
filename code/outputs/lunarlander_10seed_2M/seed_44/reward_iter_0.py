def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic dimension-agnostic transition features
    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large jumps, encourages smooth transitions)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Feature 4: Contact bonus (from info if available)
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = 0.5 * info['contact']
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on exploration and reducing absolute values
    # Mid training: balance between smoothness and goal achievement
    # Late training: fine-tune with small adjustments
    
    # Sigmoid-like transition based on training_progress
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Component weights for each stage
    abs_change_weight = 2.0 * early_weight + 0.5 * mid_weight + 0.1 * late_weight
    squared_change_weight = 0.1 * early_weight + 0.5 * mid_weight + 1.0 * late_weight
    action_cost_weight = 0.5 * early_weight + 0.3 * mid_weight + 0.1 * late_weight
    contact_weight = 0.0 * early_weight + 0.5 * mid_weight + 1.0 * late_weight
    
    # Compute reward components
    reward_abs = abs_change_weight * abs_change
    reward_smooth = -squared_change_weight * squared_change
    reward_action = -action_cost_weight * action_cost
    reward_contact = contact_weight * contact_bonus
    
    # Combine components
    reward = reward_abs + reward_smooth + reward_action + reward_contact
    
    return reward