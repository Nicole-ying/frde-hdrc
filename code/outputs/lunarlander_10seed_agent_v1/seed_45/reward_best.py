def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features - dimension agnostic
    # Feature 1: Movement toward center (reduction in absolute values)
    center_progress = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))
    
    # Feature 2: Smoothness of transition (small changes are better)
    transition_smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Velocity damping (encourage reducing speed)
    velocity_damping = -sum(abs(n[i]) for i in range(2, 4)) if len(n) >= 4 else 0.0
    
    # Feature 4: Action cost (penalize large actions)
    action_cost = -0.01 * float(action)
    
    # Feature 5: Contact bonus (from info or observation)
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = sum(n[6:8]) * 0.5
    
    # Stage weights based on training_progress
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)
    stage2_weight = 1.0 - abs(training_progress - 0.5) * 2.0
    stage3_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage-adaptive weights
    reward = (
        stage1_weight * (center_progress * 0.5 + contact_bonus * 0.3) +
        stage2_weight * (transition_smoothness * 0.3 + velocity_damping * 0.2) +
        stage3_weight * (contact_bonus * 0.4 + center_progress * 0.3) +
        action_cost * (1.0 - training_progress * 0.5)
    )
    
    return reward