def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - sum of absolute value changes
    # Positive when absolute values decrease (more centered/stable)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness - negative of squared differences (penalize large jumps)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Component 3: Action penalty - discourage excessive action usage
    # Normalize action to [0,1] range for discrete actions
    action_cost = -0.01 * float(action)
    
    # Component 4: Contact bonus - if ground contact indicators change
    # Last two dimensions are contact indicators (indices 6 and 7)
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = sum(n[6:8]) * 0.5  # Reward for ground contact
    
    # Stage-based weighting
    # Early stage: focus on smoothness and stability
    # Middle stage: balance all components
    # Late stage: emphasize contact and stability
    
    if training_progress < 0.3:
        # Early training: prioritize smooth movement and stability
        w_abs = 0.3
        w_smooth = 0.5
        w_action = 0.1
        w_contact = 0.1
    elif training_progress < 0.7:
        # Middle training: balanced approach
        w_abs = 0.3
        w_smooth = 0.3
        w_action = 0.2
        w_contact = 0.2
    else:
        # Late training: emphasize contact and fine control
        w_abs = 0.2
        w_smooth = 0.2
        w_action = 0.1
        w_contact = 0.5
    
    # Combine components
    reward = (w_abs * abs_change + 
              w_smooth * smoothness + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    return reward