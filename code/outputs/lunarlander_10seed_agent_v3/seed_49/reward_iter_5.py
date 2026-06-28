def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability (reducing absolute values)
    stability_progress = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))
    
    # Component 2: Smoothness of transition (small changes are better)
    transition_smoothness = -sum((n[i] - o[i]) ** 2 for i in range(min(6, len(o))))
    
    # Component 3: Action cost (penalize large actions)
    action_cost = -0.01 * float(action)
    
    # Component 4: Ground contact bonus (from info or obs)
    ground_contact = 0.0
    if len(o) >= 8:
        ground_contact = o[6] + o[7]  # leg contact indicators
    
    # Stage weights based on training progress
    # Early: focus on stability and smoothness
    # Middle: balance all components
    # Late: focus on ground contact and fine control
    if training_progress < 0.3:
        w_stability = 1.0
        w_smoothness = 0.5
        w_action = 0.1
        w_contact = 0.0
    elif training_progress < 0.7:
        w_stability = 0.7
        w_smoothness = 0.7
        w_action = 0.3
        w_contact = 0.5
    else:
        w_stability = 0.3
        w_smoothness = 0.3
        w_action = 0.5
        w_contact = 2.0
    
    # Combine components
    reward = (w_stability * stability_progress + 
              w_smoothness * transition_smoothness + 
              w_action * action_cost + 
              w_contact * ground_contact)
    
    return reward