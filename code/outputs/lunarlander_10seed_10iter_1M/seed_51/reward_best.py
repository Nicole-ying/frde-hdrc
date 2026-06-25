def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages moving towards desirable states)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large jumps, encourages smooth transitions)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (discourages excessive action usage)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.1
    elif action in [1, 3]:  # Side engines
        action_cost = 0.05
    
    # Feature 4: Ground contact bonus (from info or next_obs)
    ground_contact = 0.0
    if len(n) >= 8:
        ground_contact = n[6] + n[7]  # Sum of leg contact indicators
    
    # Stage-based weights that evolve with training progress
    stage1_weight = 1.0 - training_progress  # Early stage: focus on exploration
    stage2_weight = training_progress  # Later stage: focus on exploitation
    
    # Combine components with stage weights
    reward = (
        stage1_weight * (abs_diff * 0.5 - sq_change * 0.1 - action_cost * 0.3) +
        stage2_weight * (ground_contact * 2.0 - action_cost * 0.5)
    )
    
    return reward