def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays as generic lists for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Component 1: Movement magnitude change (encourages exploration/change)
    # Sum of absolute differences between current and next observation
    movement = sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Component 2: State magnitude reduction (encourages reaching stable states)
    # Compare sum of absolute values before and after - directional toward zero
    state_mag_reduction = sum(abs(o[i]) for i in range(len(o))) - sum(abs(n[i]) for i in range(len(o)))
    
    # Component 3: Action penalty (discourages excessive actuation)
    # Small penalty for taking any action
    action_cost = -0.01 * (1.0 if action != 0 else 0.0)
    
    # Component 4: Smoothness bonus (reward small changes between steps)
    # Negative squared difference to penalize large jumps
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o))) * 0.001
    
    # Component 5: Velocity penalty - penalize high speeds (directional toward zero)
    # Use the observation dimensions that correspond to velocity (indices 2 and 3 in typical LunarLander)
    # Since we don't know exact indices, use generic approach: penalize large absolute values
    # This encourages the agent to slow down, which is critical for landing
    velocity_penalty = -sum(abs(n[i]) for i in range(len(o))) * 0.01
    
    # Component 6: Contact bonus - reward ground contact (from info or obs)
    # In LunarLander, legs are at indices 6 and 7 in the observation
    # Use generic approach: check if any observation values are exactly 1.0 (binary contact indicators)
    contact_bonus = 0.0
    for i in range(len(o)):
        if abs(o[i] - 1.0) < 0.01 and abs(n[i] - 1.0) < 0.01:
            contact_bonus += 0.5  # Reward sustained contact
    
    # Stage-based weighting using training_progress (0.0 to 1.0)
    # Early stage: emphasize exploration and movement
    # Middle stage: balance exploration with stability
    # Late stage: emphasize stability, smoothness, and landing
    
    # Sigmoid-like transition for weights
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = max(0.0, 1.0 - abs(2.0 * training_progress - 1.0))
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage-appropriate weights
    # Early: explore with movement and light action cost
    # Mid: start reducing state magnitude and penalizing velocity
    # Late: strongly reward stability, smoothness, and contact
    reward = (
        early_weight * (movement * 0.1 + action_cost) +
        mid_weight * (state_mag_reduction * 0.3 + smoothness * 0.5 + velocity_penalty * 0.5 + contact_bonus * 0.2) +
        late_weight * (state_mag_reduction * 0.8 + smoothness * 1.0 + velocity_penalty * 1.0 + contact_bonus * 1.0)
    )
    
    return reward