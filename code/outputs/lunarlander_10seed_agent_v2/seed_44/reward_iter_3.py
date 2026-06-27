def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability (reducing absolute values)
    stability_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness of transition (small changes are better)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Component 3: Action cost (penalize large actions)
    action_cost = -abs(action - 1.5)  # Discrete action 0-3, center at 1.5
    
    # Component 4: Contact bonus (from info if available)
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = info['contact']
    
    # Stage weights based on training_progress
    # Early: focus on stability and smoothness
    # Middle: balance all components
    # Late: focus on fine control
    if training_progress < 0.3:
        w_stability = 1.0
        w_smoothness = 0.5
        w_action = 0.1
        w_contact = 0.0
    elif training_progress < 0.7:
        w_stability = 0.7
        w_smoothness = 0.7
        w_action = 0.3
        w_contact = 0.2
    else:
        w_stability = 0.3
        w_smoothness = 1.0
        w_action = 0.5
        w_contact = 0.3
    
    # Combine components
    reward = (w_stability * stability_diff + 
              w_smoothness * smoothness + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    return reward