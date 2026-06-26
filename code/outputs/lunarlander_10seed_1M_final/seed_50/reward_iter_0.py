def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays as generic lists for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Component 1: Movement magnitude change (encourages exploration/change)
    # Sum of absolute differences between current and next observation
    movement = sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Component 2: State magnitude reduction (encourages reaching stable states)
    # Compare sum of absolute values before and after
    state_mag_reduction = sum(abs(o[i]) for i in range(len(o))) - sum(abs(n[i]) for i in range(len(o)))
    
    # Component 3: Action penalty (discourages excessive actuation)
    # Small penalty for taking any action
    action_cost = -0.01 * (1.0 if action != 0 else 0.0)
    
    # Component 4: Smoothness bonus (reward small changes between steps)
    # Negative squared difference to penalize large jumps
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o))) * 0.001
    
    # Stage-based weighting using training_progress (0.0 to 1.0)
    # Early stage: emphasize exploration and movement
    # Middle stage: balance exploration with stability
    # Late stage: emphasize stability and smoothness
    
    # Sigmoid-like transition for weights
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = max(0.0, 1.0 - abs(2.0 * training_progress - 1.0))
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage-appropriate weights
    reward = (
        early_weight * (movement * 0.1 + action_cost) +
        mid_weight * (state_mag_reduction * 0.5 + smoothness) +
        late_weight * (state_mag_reduction * 1.0 + smoothness * 2.0)
    )
    
    return reward