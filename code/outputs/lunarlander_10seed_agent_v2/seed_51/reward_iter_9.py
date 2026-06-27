def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large absolute changes
    # Use sum of absolute differences to detect erratic movements
    abs_diff = sum(abs(o[i] - n[i]) for i in range(len(o)))
    movement_penalty = -abs_diff * 0.01
    
    # Component 2: Encourage convergence - reward when state magnitudes decrease
    # Compare absolute values before and after
    magnitude_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    convergence_reward = magnitude_change * 0.05
    
    # Component 3: Smoothness reward - penalize large squared changes (jerkiness)
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.02
    
    # Component 4: Action cost - penalize engine usage
    # Action is discrete (0-3), map to cost
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = -0.1
    elif action in [1, 3]:  # Side engines
        action_cost = -0.05
    
    # Component 5: Contact bonus - use ground contact signals from obs
    # Indices 6 and 7 are ground contact flags
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = (n[6] + n[7]) * 0.2  # Reward for ground contact
    
    # Stage-based weighting using training_progress (0.0 to 1.0)
    # Early stage: focus on smoothness and convergence
    # Middle stage: balance movement and contact
    # Late stage: emphasize contact and stability
    
    if training_progress < 0.3:
        # Early exploration - prioritize smooth movements
        w_movement = 0.3
        w_convergence = 0.4
        w_smoothness = 0.5
        w_action = 0.3
        w_contact = 0.1
    elif training_progress < 0.7:
        # Middle stage - balance all components
        w_movement = 0.4
        w_convergence = 0.3
        w_smoothness = 0.3
        w_action = 0.4
        w_contact = 0.4
    else:
        # Late stage - focus on landing and contact
        w_movement = 0.2
        w_convergence = 0.2
        w_smoothness = 0.2
        w_action = 0.3
        w_contact = 0.7
    
    # Combine components with stage weights
    reward = (w_movement * movement_penalty + 
              w_convergence * convergence_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    # Add small constant to encourage exploration
    reward += 0.01
    
    return reward