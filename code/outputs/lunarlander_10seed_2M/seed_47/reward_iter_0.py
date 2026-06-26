def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components as generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features - dimension agnostic
    # Feature 1: Movement towards origin (sum of absolute position changes)
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))
    
    # Feature 2: Velocity reduction (sum of absolute velocity changes)
    vel_change = sum(abs(o[i]) - abs(n[i]) for i in range(2, min(6, len(o))))
    
    # Feature 3: Smoothness of transition (squared difference)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 4: Contact bonus (ground contact indicators)
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = sum(n[6:8]) - sum(o[6:8])
    
    # Feature 5: Action cost (penalize large actions)
    action_cost = -0.01 * float(action)
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on exploration and movement
    # Middle stage: balance position and velocity
    # Late stage: emphasize precision and contact
    
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)  # Early stage
    stage2_weight = 1.0 - abs(2.0 * training_progress - 1.0)  # Middle stage
    stage3_weight = max(0.0, 2.0 * training_progress - 1.0)  # Late stage
    
    # Combine components with stage weights
    reward = (
        stage1_weight * (0.5 * pos_change + 0.3 * vel_change + 0.2 * action_cost) +
        stage2_weight * (0.4 * smoothness + 0.3 * pos_change + 0.3 * vel_change) +
        stage3_weight * (0.5 * contact_bonus + 0.3 * smoothness + 0.2 * pos_change)
    )
    
    # Add small baseline to encourage exploration
    reward += 0.01 * (1.0 - training_progress)
    
    return float(reward)