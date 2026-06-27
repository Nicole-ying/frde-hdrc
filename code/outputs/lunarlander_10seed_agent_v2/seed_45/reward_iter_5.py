def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude (change in position/velocity)
    diff = n - o
    movement = sum(diff[i] * diff[i] for i in range(len(diff)))
    
    # 2. Absolute state change (reduction in absolute values suggests stabilization)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 3. Action penalty (discourage excessive action usage)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.1
    elif action in [1, 3]:  # Side engines
        action_cost = 0.05
    
    # Stage-based weights
    # Early stage: focus on movement and exploration
    # Late stage: focus on stabilization and efficiency
    early_weight = max(0.0, 1.0 - training_progress * 2.0)
    late_weight = min(1.0, training_progress * 2.0)
    
    # Component rewards
    movement_reward = movement * 0.1 * early_weight
    stabilization_reward = abs_change * 0.5 * late_weight
    action_penalty = -action_cost * (1.0 + training_progress)
    
    # Contact bonus (from info if available, otherwise 0)
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = info['contact'] * 0.2 * late_weight
    
    # Combine components
    reward = movement_reward + stabilization_reward + action_penalty + contact_bonus
    
    return reward