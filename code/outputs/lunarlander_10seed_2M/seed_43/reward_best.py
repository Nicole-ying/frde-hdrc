def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs
    
    # Generic transition features - dimension agnostic
    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large jumps, encourages smooth transitions)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small action cost to encourage efficiency)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0  # no action
    elif action == 1 or action == 3:
        action_cost = 0.1  # side engines
    elif action == 2:
        action_cost = 0.2  # main engine
    
    # Feature 4: Contact bonus (from info or observation)
    # Last two dimensions of obs are ground contact indicators
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = (n[6] + n[7]) * 0.5  # both legs on ground
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on exploration and movement
    # Middle stage: balance movement and stability
    # Late stage: focus on precision and contact
    
    if training_progress < 0.3:
        # Early training: encourage movement and exploration
        w_abs = 1.0
        w_sq = -0.5
        w_action = -0.1
        w_contact = 0.0
    elif training_progress < 0.7:
        # Middle training: balance
        w_abs = 0.5
        w_sq = -0.3
        w_action = -0.2
        w_contact = 0.5
    else:
        # Late training: precision and landing
        w_abs = 0.2
        w_sq = -0.1
        w_action = -0.3
        w_contact = 1.0
    
    # Combine components
    reward = (w_abs * abs_change + 
              w_sq * sq_change + 
              w_action * action_cost + 
              w_contact * contact_bonus)
    
    return reward