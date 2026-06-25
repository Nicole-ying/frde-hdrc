def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages moving towards desirable states)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large jumps, encourages smooth transitions)
    squared_change = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small action cost to encourage efficiency)
    action_cost = -0.01 * float(action)
    
    # Feature 4: Ground contact bonus (from info if available, else from obs)
    ground_contact = 0.0
    if len(o) >= 8:
        ground_contact = o[6] + o[7]  # leg contact indicators
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on exploration and getting to good states
    # Middle stage: balance between state quality and smoothness
    # Late stage: maximize final performance
    
    # Sigmoid-like progression
    stage1_weight = 1.0 - training_progress  # Early exploration
    stage2_weight = 4.0 * training_progress * (1.0 - training_progress)  # Middle refinement
    stage3_weight = training_progress  # Late exploitation
    
    # Combine components with stage weights
    reward = (
        stage1_weight * abs_change * 0.1 +  # Encourage moving towards good states
        stage2_weight * squared_change * 0.01 +  # Encourage smooth transitions
        stage3_weight * ground_contact * 0.5 +  # Reward landing/contact
        action_cost * 0.1  # Small action penalty
    )
    
    return reward