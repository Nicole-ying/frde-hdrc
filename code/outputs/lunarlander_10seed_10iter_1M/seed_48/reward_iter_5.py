def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement towards origin (sum of absolute value reduction)
    movement_toward_origin = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness of transition (smaller changes are better)
    transition_smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalize large actions)
    # For discrete actions, penalize non-zero actions slightly
    action_cost = 0.0
    if action != 0:
        action_cost = -0.01
    
    # Feature 4: Ground contact bonus (from info or observation)
    # Check if legs are in contact (indices 6 and 7 in observation)
    leg_contact_bonus = 0.0
    if len(o) >= 8:
        leg_contact_bonus = (n[6] + n[7]) * 0.5
    
    # Stage weights based on training_progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance smoothness and goal achievement
    # Late stage: focus on precision and stability
    
    # Sigmoid-like transition for smooth weight evolution
    early_weight = max(0.0, 1.0 - training_progress * 2.0)
    mid_weight = 1.0 - abs(training_progress - 0.5) * 2.0
    late_weight = max(0.0, training_progress * 2.0 - 1.0)
    
    # Combine components with stage-appropriate weights
    reward = (
        early_weight * movement_toward_origin * 0.3 +
        mid_weight * transition_smoothness * 0.2 +
        late_weight * leg_contact_bonus * 0.4 +
        action_cost * 0.1
    )
    
    # Add small exploration bonus in early stages
    exploration_bonus = early_weight * 0.01
    
    return reward + exploration_bonus