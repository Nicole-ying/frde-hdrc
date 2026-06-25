def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages moving towards desirable states)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (captures magnitude of state changes)
    sq_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (encourages efficient control)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.1
    elif action in [1, 3]:  # Side engines
        action_cost = 0.05
    
    # Feature 4: Ground contact bonus (from info or observation)
    ground_contact = 0.0
    if len(o) >= 8:
        ground_contact = (o[6] + o[7]) * 0.5  # Average of leg contacts
    
    # Stage-based weights that evolve with training progress
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)  # Early: focus on state changes
    stage2_weight = 1.0 - abs(1.0 - 2.0 * training_progress)  # Middle: balance
    stage3_weight = max(0.0, 2.0 * training_progress - 1.0)  # Late: focus on ground contact
    
    # Combine components with stage weights
    reward = (
        stage1_weight * (abs_diff * 0.5 - sq_diff * 0.1 - action_cost) +
        stage2_weight * (abs_diff * 0.3 - sq_diff * 0.05 + ground_contact * 0.2 - action_cost * 0.5) +
        stage3_weight * (ground_contact * 1.0 - action_cost * 0.3)
    )
    
    return reward