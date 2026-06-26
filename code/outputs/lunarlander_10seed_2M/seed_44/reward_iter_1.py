def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Feature 1: Movement toward zero (directional - encourages stabilization)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Penalize large velocity changes (smoothness)
    vel_change = sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Feature 3: Action penalty (discourage unnecessary engine use)
    action_cost = abs(action) if hasattr(action, '__iter__') else abs(action)
    
    # Feature 4: Contact bonus from info
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = float(info['contact'])
    
    # Stage-based weights
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Component weights
    abs_change_weight = 1.0 * early_weight + 0.3 * mid_weight + 0.1 * late_weight
    vel_change_weight = 0.1 * early_weight + 0.5 * mid_weight + 1.0 * late_weight
    action_cost_weight = 0.3 * early_weight + 0.2 * mid_weight + 0.1 * late_weight
    contact_weight = 0.0 * early_weight + 0.5 * mid_weight + 1.0 * late_weight
    
    # Compute reward components
    reward_abs = abs_change_weight * abs_change
    reward_smooth = -vel_change_weight * vel_change
    reward_action = -action_cost_weight * action_cost
    reward_contact = contact_weight * contact_bonus
    
    # Combine components
    reward = reward_abs + reward_smooth + reward_action + reward_contact
    
    return reward