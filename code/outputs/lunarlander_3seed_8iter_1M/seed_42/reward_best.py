def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Dimension-agnostic transition features
    # Feature 1: Movement towards center - sum of absolute position changes
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))
    
    # Feature 2: Velocity reduction - penalize high velocities
    vel_change = sum((o[i] - n[i]) ** 2 for i in range(2, min(6, len(o))))
    
    # Feature 3: Angular stability - penalize angular velocity
    ang_change = sum((o[i] - n[i]) ** 2 for i in range(4, min(6, len(o))))
    
    # Feature 4: Ground contact bonus
    ground_contact = sum(n[i] for i in range(6, min(8, len(n))))
    
    # Feature 5: Action cost - small penalty for using engines
    action_cost = 0.0
    if action in [1, 2, 3]:
        action_cost = 0.01
    
    # Stage weights based on training_progress
    early_weight = max(0.0, 1.0 - training_progress * 2.0)
    mid_weight = 1.0 - abs(training_progress - 0.5) * 2.0
    late_weight = max(0.0, training_progress * 2.0 - 1.0)
    
    # Combine features with stage weights
    reward = (
        early_weight * pos_change * 0.5 +
        mid_weight * (-vel_change * 0.1 - ang_change * 0.1 + ground_contact * 2.0) +
        late_weight * (ground_contact * 3.0 - action_cost * 0.5)
    )
    
    return reward