def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays for generic processing
    o = obs
    n = next_obs
    
    # Component 1: Encourages reduction in absolute state magnitudes (stabilization)
    # Positive when state values move toward zero
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    stability_reward = abs_diff / len(o)  # Normalize by dimension count
    
    # Component 2: Penalizes large state changes (smoothness)
    # Smaller squared differences indicate smoother transitions
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff / (len(o) * 10.0)  # Normalize and scale
    
    # Component 3: Small action penalty to encourage efficiency
    # Discrete action 0-3, penalize higher values slightly
    action_cost = -0.01 * float(action)
    
    # Component 4: Use ground contact signals from info if available
    # Positive reward for ground contact (landing legs)
    contact_bonus = 0.0
    if 'leg_contact' in info:
        contact_bonus = 0.5 * float(info['leg_contact'])
    elif len(o) >= 8:
        # Last two obs dimensions are leg contact indicators
        contact_bonus = 0.5 * (o[6] + o[7])
    
    # Stage weights based on training_progress (0.0 to 1.0)
    # Early training: focus on stability and smoothness
    # Mid training: maintain stability, add contact awareness
    # Late training: fine-tune with all components
    
    if training_progress < 0.3:
        # Early stage: prioritize stabilization
        w_stability = 1.0
        w_smoothness = 0.5
        w_action = 0.1
        w_contact = 0.0
    elif training_progress < 0.7:
        # Mid stage: balance stability with contact seeking
        w_stability = 0.7
        w_smoothness = 0.3
        w_action = 0.2
        w_contact = 0.5
    else:
        # Late stage: refine all aspects
        w_stability = 0.5
        w_smoothness = 0.2
        w_action = 0.3
        w_contact = 0.8
    
    # Combine components with stage weights
    reward = (w_stability * stability_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    return reward