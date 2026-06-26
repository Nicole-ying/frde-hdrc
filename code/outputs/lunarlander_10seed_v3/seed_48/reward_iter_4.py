def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - reward reduction in absolute values
    # This is directional: rewards moving state values toward zero
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(min(len(o), len(n))))
    movement_reward = abs_change * 0.8
    
    # Component 2: Smoothness - penalize large changes between steps
    diff_sq = sum((n[i] - o[i]) ** 2 for i in range(min(len(o), len(n))))
    smoothness_penalty = -diff_sq * 0.02
    
    # Component 3: Action cost - penalize engine usage
    action_cost = -0.005 * float(action)
    
    # Component 4: Contact bonus - reward ground contact (from obs)
    contact_bonus = 0.0
    if len(o) >= 8 and len(n) >= 8:
        # Last two dimensions are ground contact indicators
        curr_contact = n[-2] + n[-1]
        contact_bonus = curr_contact * 1.5  # Reward being in contact, not just change
    
    # Component 5: Velocity penalty - penalize high speed (discourage crashing)
    vel_penalty = 0.0
    if len(o) >= 4 and len(n) >= 4:
        curr_speed = abs(n[2]) + abs(n[3])
        vel_penalty = -curr_speed * 0.5  # Penalize current speed directly
    
    # Component 6: Angle stability - penalize angular velocity and angle
    angle_penalty = 0.0
    if len(o) >= 6 and len(n) >= 6:
        # Penalize large angle and angular velocity
        angle_penalty = -abs(n[4]) * 1.0 - abs(n[5]) * 0.6
    
    # Stage-based weights that evolve with training_progress
    # Early training: focus on movement and exploration
    # Mid training: balance smoothness and contact
    # Late training: refine stability
    
    stage1_weight = max(0.0, 1.0 - training_progress * 2.0)  # Decreases from 1 to 0
    stage2_weight = 1.0 - abs(training_progress - 0.5) * 2.0  # Peaks at 0.5
    stage3_weight = max(0.0, training_progress * 2.0 - 1.0)  # Increases from 0 to 1
    
    # Combine components with stage weights
    reward = (
        stage1_weight * movement_reward +
        stage2_weight * smoothness_penalty +
        stage3_weight * contact_bonus +
        stage2_weight * vel_penalty +
        stage3_weight * angle_penalty +
        action_cost * (1.0 - stage3_weight * 0.5)  # Reduce action cost importance late
    )
    
    return reward