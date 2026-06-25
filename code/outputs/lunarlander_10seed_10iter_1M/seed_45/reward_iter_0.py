def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large absolute changes
    # Use absolute difference sum as a generic stability signal
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    stability_reward = abs_diff * 0.1
    
    # Component 2: Smoothness - penalize large squared changes
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.05
    
    # Component 3: Action cost - penalize taking actions (encourage efficiency)
    # For discrete actions, penalize non-zero actions (action 0 is typically no-op)
    action_cost = -0.01 if action != 0 else 0.0
    
    # Component 4: Contact bonus - reward ground contact from info if available
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = 0.5 if info['contact'] else 0.0
    
    # Stage-based weighting
    # Early training: focus on stability and smoothness
    # Mid training: balance all components
    # Late training: emphasize contact and efficiency
    if training_progress < 0.3:
        w_stability = 1.0
        w_smoothness = 0.5
        w_action = 0.1
        w_contact = 0.0
    elif training_progress < 0.7:
        w_stability = 0.6
        w_smoothness = 0.8
        w_action = 0.3
        w_contact = 0.5
    else:
        w_stability = 0.3
        w_smoothness = 0.3
        w_action = 0.5
        w_contact = 1.0
    
    # Combine components
    reward = (w_stability * stability_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    return reward