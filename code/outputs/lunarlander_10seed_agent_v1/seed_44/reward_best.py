def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays from observations
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large absolute changes
    # This encourages smooth transitions and penalizes erratic behavior
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_change * 0.1
    
    # Component 2: Smoothness - penalize large squared differences
    # This encourages the agent to make small, controlled changes
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.5
    
    # Component 3: Action cost - penalize taking actions to encourage efficiency
    # Convert discrete action to a small cost
    action_cost = -0.01 * float(action)
    
    # Component 4: Contact bonus - reward ground contact from info if available
    contact_bonus = 0.0
    if info and isinstance(info, dict):
        # Look for any contact-related signals in info
        for key, value in info.items():
            if isinstance(value, (int, float)) and value > 0:
                contact_bonus += value * 0.05
    
    # Component 5: Terminal state bonus - reward reaching stable states
    # Check if next_obs indicates a stable configuration (small velocities, upright)
    stability_bonus = 0.0
    if len(n) >= 6:
        # Penalize high velocities (indices 2, 3 for velocity components)
        vel_penalty = -(abs(n[2]) + abs(n[3])) * 0.2
        # Reward being upright (index 4 for angle, small angle is good)
        angle_penalty = -abs(n[4]) * 0.1
        stability_bonus = vel_penalty + angle_penalty
    
    # Stage-based weighting using training_progress
    # Early training: focus on exploration and movement
    # Mid training: balance smoothness and stability
    # Late training: emphasize stability and efficiency
    
    # Linear interpolation between stages
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage weights
    reward = (
        early_weight * (movement_reward * 0.3 + action_cost * 0.7) +
        mid_weight * (smoothness_penalty * 0.4 + stability_bonus * 0.4 + contact_bonus * 0.2) +
        late_weight * (stability_bonus * 0.5 + smoothness_penalty * 0.3 + contact_bonus * 0.2)
    )
    
    # Add a small constant to encourage exploration
    exploration_bonus = 0.01 * (1.0 - training_progress)
    
    return reward + exploration_bonus