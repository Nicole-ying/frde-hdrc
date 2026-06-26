def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement magnitude - sum of squared differences
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 2: Absolute value change - encourages moving toward zero
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 3: Action penalty - discourage excessive action usage
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = -0.5
    elif action in [1, 3]:  # Side engines
        action_cost = -0.2
    
    # Feature 4: Contact bonus from info (if available)
    contact_bonus = 0.0
    if info and 'contact' in info:
        contact_bonus = 0.1 * info['contact']
    
    # Stage weights based on training_progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance movement with stability
    # Late stage: emphasize precision and contact
    
    if training_progress < 0.3:
        # Early exploration
        w_movement = 1.0
        w_abs_change = 0.5
        w_action = 0.3
        w_contact = 0.0
    elif training_progress < 0.7:
        # Middle stage - balance
        w_movement = 0.7
        w_abs_change = 0.8
        w_action = 0.5
        w_contact = 0.3
    else:
        # Late stage - precision
        w_movement = 0.3
        w_abs_change = 1.0
        w_action = 0.7
        w_contact = 0.6
    
    # Combine components
    reward = (w_movement * movement + 
              w_abs_change * abs_change + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    # Ensure numerical stability with a small offset
    reward = reward + 0.01
    
    return reward