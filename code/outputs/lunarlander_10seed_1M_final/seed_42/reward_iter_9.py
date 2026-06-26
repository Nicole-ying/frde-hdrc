def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards center - encourage reducing absolute values
    movement_reward = 0.0
    for i in range(min(4, len(o))):
        movement_reward += abs(o[i]) - abs(n[i])
    
    # Component 2: Smoothness - penalize large changes
    smoothness_penalty = 0.0
    for i in range(min(4, len(o))):
        smoothness_penalty += (n[i] - o[i]) ** 2
    
    # Component 3: Action cost - penalize engine usage
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # Component 4: Ground contact bonus
    ground_contact_bonus = 0.0
    if len(n) >= 8:
        ground_contact_bonus = n[6] + n[7]  # Sum of leg contact indicators
    
    # Stage weights based on training progress
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)
    stage2_weight = 1.0 - abs(1.0 - 2.0 * training_progress)
    stage3_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage-appropriate weights
    reward = 0.0
    
    # Stage 1 (early): Focus on movement and exploration
    reward += stage1_weight * (movement_reward * 2.0 - smoothness_penalty * 0.1 - action_cost)
    
    # Stage 2 (middle): Balance all components
    reward += stage2_weight * (movement_reward * 1.5 - smoothness_penalty * 0.2 - action_cost * 0.5 + ground_contact_bonus * 0.5)
    
    # Stage 3 (late): Focus on stability and landing
    reward += stage3_weight * (-smoothness_penalty * 0.5 - action_cost * 0.3 + ground_contact_bonus * 2.0)
    
    return reward