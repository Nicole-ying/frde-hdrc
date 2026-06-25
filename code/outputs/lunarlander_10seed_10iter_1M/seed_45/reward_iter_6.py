def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Sum of absolute differences (encourages movement toward target states)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change magnitude (penalizes large erratic changes)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalizes fuel usage)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 1.0
    elif action in [1, 3]:  # Side engines
        action_cost = 0.5
    
    # Feature 4: Ground contact bonus (from info or observation)
    ground_contact = 0.0
    if len(o) >= 8:
        ground_contact = n[6] + n[7]  # Sum of leg contact indicators
    
    # Stage weights based on training_progress
    # Early stage: focus on exploration and movement
    # Middle stage: balance movement and stability
    # Late stage: prioritize precision and landing
    
    # Smooth stage transition
    early_weight = max(0.0, 1.0 - training_progress * 2.0)
    middle_weight = 1.0 - abs(training_progress - 0.5) * 2.0
    late_weight = max(0.0, training_progress * 2.0 - 1.0)
    
    # Component weights that evolve with training
    w_abs_diff = 0.5 * early_weight + 0.3 * middle_weight + 0.1 * late_weight
    w_sq_change = -0.1 * early_weight - 0.3 * middle_weight - 0.5 * late_weight
    w_action_cost = -0.2 * early_weight - 0.2 * middle_weight - 0.1 * late_weight
    w_ground_contact = 0.0 * early_weight + 0.2 * middle_weight + 0.5 * late_weight
    
    # Compute reward components
    reward_abs_diff = w_abs_diff * abs_diff
    reward_sq_change = w_sq_change * sq_change
    reward_action_cost = w_action_cost * action_cost
    reward_ground_contact = w_ground_contact * ground_contact
    
    # Combine components
    reward = reward_abs_diff + reward_sq_change + reward_action_cost + reward_ground_contact
    
    return reward