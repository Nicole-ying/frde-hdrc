def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement towards center - sum of absolute position changes
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Feature 2: Velocity reduction - penalize high velocities
    vel_change = sum((n[i] - o[i]) ** 2 for i in range(2, 4))
    
    # Feature 3: Angle stabilization - encourage upright orientation
    angle_diff = abs(n[4]) - abs(o[4])
    
    # Feature 4: Angular velocity damping
    ang_vel_change = abs(n[5]) - abs(o[5])
    
    # Feature 5: Ground contact bonus
    ground_contact = sum(n[6:8])
    
    # Feature 6: Action penalty (small cost for using engines)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = -0.1
    elif action in [1, 3]:  # Side engines
        action_cost = -0.05
    
    # Stage-based weights that evolve with training progress
    # Stage 1 (early): Focus on position and velocity
    w1 = 1.0 - 0.5 * training_progress  # Position weight decreases
    w2 = 0.5 + 0.5 * training_progress   # Velocity weight increases
    
    # Stage 2 (mid): Introduce angle and angular velocity
    w3 = 0.3 * training_progress          # Angle weight grows
    w4 = 0.2 * training_progress          # Angular velocity weight grows
    
    # Stage 3 (late): Ground contact becomes important
    w5 = 0.5 * max(0, training_progress - 0.5) * 2  # Ground contact weight in second half
    
    # Action penalty is always small
    w6 = 0.1
    
    # Compute reward components
    reward_pos = w1 * pos_change
    reward_vel = -w2 * vel_change * 0.1  # Scale down velocity penalty
    reward_angle = w3 * (-angle_diff)     # Negative because we want angle to decrease
    reward_ang_vel = w4 * (-ang_vel_change)  # Negative because we want angular velocity to decrease
    reward_ground = w5 * ground_contact
    reward_action = w6 * action_cost
    
    # Combine all components
    reward = reward_pos + reward_vel + reward_angle + reward_ang_vel + reward_ground + reward_action
    
    return reward