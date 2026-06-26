def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Component 1: Directional movement toward center (abs_change)
    # Positive when moving toward zero (desirable)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness penalty - penalize large state changes
    # Use L1 norm instead of squared to reduce scale and avoid over-penalizing
    smoothness = sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Component 3: Action cost - penalize engine use
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 1.0
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # Component 4: Leg contact bonus - reward ground contact
    leg_contact = o[6] + o[7]
    
    # Component 5: Velocity penalty - discourage high speeds (indices 2,3)
    vel_penalty = abs(o[2]) + abs(o[3])
    
    # Component 6: Angle penalty - discourage tilting (index 4)
    angle_penalty = abs(o[4])
    
    # Component 7: Angular velocity penalty - discourage spinning (index 5)
    ang_vel_penalty = abs(o[5])
    
    # Stage-based weights - early exploration, later stabilization
    w1 = 1.0 - 0.3 * training_progress  # abs_change weight decreases
    w2 = 0.05 + 0.15 * training_progress  # smoothness penalty increases
    w3 = 0.1 * (1.0 - training_progress)  # action cost decreases
    w4 = 0.3 * training_progress  # leg contact bonus increases
    w5 = 0.2 * (1.0 - training_progress)  # velocity penalty decreases
    w6 = 0.3 * (1.0 - training_progress)  # angle penalty decreases
    w7 = 0.1 * (1.0 - training_progress)  # angular velocity penalty decreases
    
    # Combine components
    reward = (w1 * abs_change
              - w2 * smoothness
              - w3 * action_cost
              + w4 * leg_contact
              - w5 * vel_penalty
              - w6 * angle_penalty
              - w7 * ang_vel_penalty)
    
    return reward