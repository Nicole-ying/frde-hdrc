def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude (encourage change in state)
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 2. Absolute position change (encourage reaching new states)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 3. Action penalty (discourage excessive action)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # 4. Contact bonus (from info or observation)
    contact_bonus = 0.0
    if len(o) >= 8:
        # Last two elements are ground contact indicators
        contact_bonus = (o[6] + o[7]) * 0.5
    
    # Stage-based weighting
    stage1_end = 0.3
    stage2_end = 0.7
    
    if training_progress < stage1_end:
        # Early stage: focus on exploration and movement
        w_movement = 1.0
        w_abs_change = 0.5
        w_action = -0.2
        w_contact = 0.1
    elif training_progress < stage2_end:
        # Middle stage: balance exploration with control
        w_movement = 0.7
        w_abs_change = 0.3
        w_action = -0.4
        w_contact = 0.3
    else:
        # Late stage: focus on stability and contact
        w_movement = 0.3
        w_abs_change = 0.1
        w_action = -0.6
        w_contact = 0.8
    
    # Compute reward components
    reward = (w_movement * movement + 
              w_abs_change * abs_change + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    # Add small exploration bonus for new states
    exploration_bonus = 0.01 * sum(abs(n[i] - o[i]) for i in range(len(o)))
    reward += exploration_bonus * (1.0 - training_progress)
    
    return float(reward)