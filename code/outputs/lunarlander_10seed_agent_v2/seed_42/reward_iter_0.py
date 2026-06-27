def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement magnitude - encourage reducing absolute state values (stabilization)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness - penalize large changes in state (encourage gentle control)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost - penalize engine usage
    action_cost = -0.01 * float(action)
    
    # Feature 4: Contact bonus - encourage ground contact (from last two obs dimensions)
    contact_bonus = 0.0
    if len(o) >= 2:
        contact_bonus = n[-1] + n[-2]  # ground contact indicators
    
    # Stage-based weighting
    # Early stage: focus on stabilization and smoothness
    # Middle stage: balance stabilization with contact seeking
    # Late stage: emphasize contact and fine control
    
    if training_progress < 0.3:
        # Early exploration - prioritize reducing state magnitudes and smoothness
        w_abs = 1.0
        w_smooth = 0.5
        w_action = 0.1
        w_contact = 0.0
    elif training_progress < 0.7:
        # Middle stage - transition to contact seeking
        w_abs = 0.6
        w_smooth = 0.3
        w_action = 0.2
        w_contact = 0.5
    else:
        # Late stage - focus on landing (contact) with fine control
        w_abs = 0.3
        w_smooth = 0.2
        w_action = 0.3
        w_contact = 1.0
    
    # Combine components
    reward = (w_abs * abs_diff + 
              w_smooth * smoothness + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    # Scale to reasonable range
    reward = reward * 0.1
    
    return reward