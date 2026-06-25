def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement toward center - penalize absolute values increasing
    movement_penalty = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))
    
    # Feature 2: Smoothness - penalize large changes in observations
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost - penalize taking actions (encourage efficiency)
    action_cost = -0.01 * abs(action - 1.5)  # Discrete action penalty
    
    # Feature 4: Ground contact bonus (from info or observation indices 6,7)
    ground_contact = 0.0
    if len(o) >= 8:
        ground_contact = (n[6] + n[7]) * 0.5  # Average of leg contact indicators
    
    # Feature 5: Stability - penalize angular velocity and angle
    stability = 0.0
    if len(o) >= 6:
        angle_penalty = -abs(n[4]) * 0.1  # Penalize angle
        angular_vel_penalty = -abs(n[5]) * 0.05  # Penalize angular velocity
        stability = angle_penalty + angular_vel_penalty
    
    # Stage-based weights that evolve with training_progress
    # Early training: focus on movement and exploration
    # Mid training: balance movement, smoothness, and stability
    # Late training: emphasize stability and ground contact
    
    stage1_weight = max(0.0, 1.0 - training_progress * 2.0)  # Decays in first half
    stage2_weight = 1.0 - abs(training_progress - 0.5) * 2.0  # Peaks at middle
    stage3_weight = max(0.0, training_progress * 2.0 - 1.0)  # Grows in second half
    
    # Combine components with stage weights
    reward = (
        stage1_weight * movement_penalty * 0.5 +
        stage2_weight * smoothness * 0.3 +
        stage1_weight * action_cost * 0.2 +
        stage3_weight * ground_contact * 2.0 +
        stage3_weight * stability * 0.5 +
        stage2_weight * (movement_penalty + smoothness) * 0.1
    )
    
    return reward