def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Feature 1: Movement toward zero (directional stabilization)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness - directional form, reward reducing change magnitude
    smoothness = -sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Feature 3: Action cost - scale for discrete actions (0-3)
    action_cost = -0.1 * float(action)
    
    # Feature 4: Contact bonus - normalize to [0,2] range
    contact_bonus = 0.0
    if len(o) >= 2:
        contact_bonus = n[-1] + n[-2]
    
    # Feature 5: Angular velocity penalty (from obs index 5)
    ang_vel_penalty = 0.0
    if len(o) >= 6:
        ang_vel_penalty = -abs(n[5])
    
    # Stage-based weighting
    if training_progress < 0.3:
        # Early: focus on stabilization and smooth movement
        w_abs = 1.0
        w_smooth = 0.3
        w_action = 0.05
        w_contact = 0.0
        w_ang = 0.2
    elif training_progress < 0.7:
        # Middle: balance all signals
        w_abs = 0.5
        w_smooth = 0.2
        w_action = 0.1
        w_contact = 0.3
        w_ang = 0.3
    else:
        # Late: focus on landing with stability
        w_abs = 0.3
        w_smooth = 0.1
        w_action = 0.15
        w_contact = 0.8
        w_ang = 0.4
    
    # Combine components
    reward = (w_abs * abs_diff + 
              w_smooth * smoothness + 
              w_action * action_cost + 
              w_contact * contact_bonus +
              w_ang * ang_vel_penalty)
    
    # Scale to reasonable range
    reward = reward * 0.1
    
    return reward