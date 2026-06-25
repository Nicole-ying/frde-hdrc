def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large sudden changes)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Feature 4: Progress-based bonus for consistent improvement
    progress_bonus = training_progress * 0.1
    
    # Stage weights that evolve with training progress
    # Early training: focus on reducing absolute values
    # Mid training: balance between movement and stability
    # Late training: fine-tune with small adjustments
    w1 = 1.0 - 0.5 * training_progress  # weight for abs_change
    w2 = 0.1 + 0.4 * training_progress   # weight for squared_change
    w3 = 0.1 * (1.0 - training_progress) # weight for action_cost
    w4 = 0.05 * training_progress        # weight for progress_bonus
    
    # Combine components
    reward = (
        w1 * abs_change
        - w2 * squared_change
        - w3 * action_cost
        + w4 * progress_bonus
    )
    
    return reward