def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Dimension-agnostic transition features
    
    # 1. Movement magnitude: sum of squared differences between next and current observations
    # Captures any change in state, encouraging exploration
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 2. Absolute state change: sum of absolute differences
    # Captures directional changes, penalizes large deviations
    abs_change = sum(abs(n[i]) - abs(o[i]) for i in range(len(o)))
    
    # 3. Action penalty: small cost for taking actions to prevent thrashing
    # Discrete actions 0,1,2,3 - penalize any action slightly
    action_cost = 0.01 * (1.0 if action != 0 else 0.0)
    
    # 4. Contact bonus: check if legs are making contact (indices 6 and 7)
    # Rewards landing/ground contact
    left_leg_contact = n[6] if len(n) > 6 else 0.0
    right_leg_contact = n[7] if len(n) > 7 else 0.0
    contact_bonus = 0.5 * (left_leg_contact + right_leg_contact)
    
    # 5. Stability bonus: reward low angular velocity and angle
    # Index 4 is angle, index 5 is angular velocity
    angle_penalty = -0.1 * abs(n[4]) if len(n) > 4 else 0.0
    angular_vel_penalty = -0.05 * abs(n[5]) if len(n) > 5 else 0.0
    
    # Stage-based weighting using training_progress
    # Early training: focus on exploration and movement
    # Late training: focus on stability and contact
    
    # Sigmoid-like transition based on training_progress
    progress = max(0.0, min(1.0, training_progress))
    
    # Exploration weight decreases with progress
    exploration_weight = 1.0 - 0.8 * progress
    
    # Stability weight increases with progress
    stability_weight = 0.2 + 0.8 * progress
    
    # Contact weight increases with progress
    contact_weight = 0.1 + 0.9 * progress
    
    # Combine components
    reward = (
        exploration_weight * (0.5 * movement + 0.3 * abs_change - action_cost)
        + stability_weight * (angle_penalty + angular_vel_penalty)
        + contact_weight * contact_bonus
    )
    
    return reward