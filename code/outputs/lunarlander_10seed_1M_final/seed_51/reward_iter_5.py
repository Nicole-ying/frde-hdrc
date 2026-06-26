def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement magnitude - encourages change in state
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 2: Absolute value reduction - encourages moving toward zero
    abs_reduction = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 3: Action penalty - discourage excessive actions
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.2
    
    # Feature 4: Contact bonus from info (if available)
    contact_bonus = 0.0
    if info and 'contact' in info:
        contact_bonus = float(info['contact']) * 0.5
    
    # Stage weights based on training_progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance movement with action efficiency
    # Late stage: emphasize stability and contact
    
    # Sigmoid-like stage transitions
    early_weight = max(0.0, 1.0 - training_progress * 2.0)
    mid_weight = 1.0 - abs(training_progress - 0.5) * 2.0
    late_weight = max(0.0, training_progress * 2.0 - 1.0)
    
    # Component weights that evolve with training
    movement_weight = 0.3 + 0.2 * early_weight
    abs_reduction_weight = 0.4 + 0.2 * mid_weight
    action_penalty_weight = 0.1 + 0.2 * late_weight
    contact_weight = 0.2 * late_weight
    
    # Compute reward components
    movement_reward = movement * movement_weight
    abs_reduction_reward = abs_reduction * abs_reduction_weight
    action_penalty = -action_cost * action_penalty_weight
    contact_reward = contact_bonus * contact_weight
    
    # Combine components
    reward = movement_reward + abs_reduction_reward + action_penalty + contact_reward
    
    # Add small exploration bonus for early stages
    exploration_bonus = 0.01 * early_weight * sum(abs(n[i] - o[i]) for i in range(len(o)))
    reward += exploration_bonus
    
    return reward