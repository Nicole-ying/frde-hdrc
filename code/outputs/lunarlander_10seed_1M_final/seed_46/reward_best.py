def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement toward zero - sum of absolute value reduction
    abs_reduction = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness - squared change magnitude (penalize large jumps)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost - penalize action usage
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # Feature 4: Contact signals from info (if available)
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = float(info['contact']) * 0.5
    elif len(o) >= 8:
        # Use last two observation dimensions as contact indicators
        contact_bonus = (o[6] + o[7]) * 0.5
    
    # Stage-based weights that evolve with training_progress
    # Stage 1 (early): Focus on reaching target area (abs_reduction)
    # Stage 2 (mid): Balance movement and smoothness
    # Stage 3 (late): Fine-tune with contact and minimal action
    
    # Sigmoid-like transition between stages
    stage1_weight = max(0.0, 1.0 - training_progress * 2.0)  # Decays from 1 to 0
    stage2_weight = 1.0 - abs(training_progress - 0.5) * 2.0  # Peaks at 0.5
    stage3_weight = max(0.0, training_progress * 2.0 - 1.0)  # Grows from 0 to 1
    
    # Normalize weights to sum to 1.0
    total_weight = stage1_weight + stage2_weight + stage3_weight
    if total_weight > 0.0:
        stage1_weight /= total_weight
        stage2_weight /= total_weight
        stage3_weight /= total_weight
    
    # Component rewards with stage-appropriate scaling
    movement_reward = abs_reduction * 0.1  # Scale to reasonable range
    smoothness_penalty = -squared_change * 0.01  # Small penalty for large changes
    action_penalty = -action_cost * 0.2  # Penalize engine usage
    contact_reward = contact_bonus * 0.3  # Reward ground contact
    
    # Combine components with stage weights
    reward = (
        stage1_weight * (movement_reward * 2.0 + contact_reward * 0.5) +
        stage2_weight * (movement_reward * 1.0 + smoothness_penalty * 0.5 + action_penalty * 0.5) +
        stage3_weight * (contact_reward * 2.0 + smoothness_penalty * 0.3 + action_penalty * 0.2)
    )
    
    # Add small exploration bonus early, decay later
    exploration_bonus = (1.0 - training_progress) * 0.01
    reward += exploration_bonus
    
    return reward