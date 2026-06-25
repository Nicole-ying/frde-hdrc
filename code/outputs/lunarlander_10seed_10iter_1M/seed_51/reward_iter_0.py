def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large absolute changes
    # Measures how much absolute values are decreasing (stabilization)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    stability_reward = abs_diff / len(o)
    
    # Component 2: Smoothness - penalize large jumps in state
    # Use squared differences normalized by dimension
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff / (len(o) * 10.0)
    
    # Component 3: Action cost - penalize engine usage
    # Discrete actions: 0=noop, 1=left, 2=main, 3=right
    # Main engine (action=2) costs more than side engines (1,3)
    if action == 2:
        action_cost = -0.5
    elif action in [1, 3]:
        action_cost = -0.25
    else:
        action_cost = 0.0
    
    # Component 4: Contact bonus - reward ground contact from legs
    # Leg contact indicators are at indices 6 and 7 in the observation
    leg_contact_bonus = 0.0
    if len(o) >= 8:
        # Check if either leg made contact (transition from 0 to 1)
        prev_contact = o[6] + o[7]
        curr_contact = n[6] + n[7]
        if curr_contact > prev_contact:
            leg_contact_bonus = 1.0
    
    # Stage-based weights that evolve with training_progress
    # Early training: focus on stability and smoothness
    # Mid training: balance all components
    # Late training: emphasize contact and fine control
    
    if training_progress < 0.3:
        # Early stage: learn basic control
        w_stability = 1.0
        w_smoothness = 0.5
        w_action = 0.3
        w_contact = 0.1
    elif training_progress < 0.7:
        # Mid stage: refine control
        w_stability = 0.7
        w_smoothness = 0.7
        w_action = 0.5
        w_contact = 0.5
    else:
        # Late stage: fine-tune landing
        w_stability = 0.5
        w_smoothness = 0.3
        w_action = 0.7
        w_contact = 1.0
    
    # Combine components with stage weights
    reward = (w_stability * stability_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * leg_contact_bonus)
    
    return reward