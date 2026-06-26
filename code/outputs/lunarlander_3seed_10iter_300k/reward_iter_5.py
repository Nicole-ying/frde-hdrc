def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement towards center (sum of absolute position changes)
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Feature 2: Smoothness - penalize large changes in state
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(8))
    
    # Feature 3: Action penalty - discourage excessive engine use
    action_cost = -0.01 * float(action)
    
    # Feature 4: Ground contact bonus
    ground_contact = sum(n[6:8])  # legs contact indicators
    
    # Stage weights based on training progress
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Combine features with stage-appropriate weights
    reward = (
        early_weight * (0.5 * pos_change + 0.3 * smoothness + 0.2 * action_cost) +
        late_weight * (0.4 * ground_contact + 0.3 * smoothness + 0.3 * action_cost)
    )
    
    return reward