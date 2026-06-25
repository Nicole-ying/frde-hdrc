def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays for generic processing
    o = obs
    n = next_obs
    
    # Feature 1: Directional movement toward zero (encourages stabilization)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness penalty - penalize large changes between steps
    # Use absolute change instead of squared to reduce scale
    smoothness = sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Feature 3: Action cost - penalize engine usage
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # Feature 4: Velocity change signal - penalize rapid velocity changes
    # Assuming obs[2] and obs[3] are velocities (x, y)
    vel_change = abs(n[2] - o[2]) + abs(n[3] - o[3])
    
    # Feature 5: Stability signal - penalize angular velocity
    # Assuming obs[5] is angular velocity
    angular_vel_penalty = abs(n[5]) if len(o) > 5 else 0.0
    
    # Feature 6: Contact bonus - reward ground contact
    # Assuming obs[6] and obs[7] are leg contact indicators
    contact_bonus = 0.0
    if len(o) > 7:
        contact_bonus = n[6] + n[7]  # Sum of leg contacts
    
    # Stage weights based on training_progress
    if training_progress < 0.3:
        # Early stage: encourage exploration and basic movement
        w_abs = 1.0
        w_smooth = -0.3
        w_action = -0.1
        w_vel = -0.2
        w_angular = -0.5
        w_contact = 0.0
    elif training_progress < 0.7:
        # Middle stage: balance exploration with stability
        w_abs = 0.7
        w_smooth = -0.5
        w_action = -0.3
        w_vel = -0.4
        w_angular = -1.0
        w_contact = 0.5
    else:
        # Late stage: focus on precision and landing
        w_abs = 0.3
        w_smooth = -0.8
        w_action = -0.5
        w_vel = -0.6
        w_angular = -2.0
        w_contact = 2.0
    
    # Combine components
    reward = (w_abs * abs_change +
              w_smooth * smoothness +
              w_action * action_cost +
              w_vel * vel_change +
              w_angular * angular_vel_penalty +
              w_contact * contact_bonus)
    
    return reward