def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # --- Signal 1: Movement toward origin (directional) ---
    # Encourages reducing absolute values of state components
    movement_toward_origin = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # --- Signal 2: Velocity penalty (from obs indices 2,3) ---
    # Penalizes high speed to prevent crashes
    # obs[2] = vel.x, obs[3] = vel.y
    vel_penalty = 0.0
    if len(o) >= 4:
        vel_penalty = (o[2] ** 2 + o[3] ** 2)  # squared velocity magnitude
    
    # --- Signal 3: Angle penalty (from obs index 4) ---
    # Penalizes large angles to prevent tumbling
    angle_penalty = 0.0
    if len(o) >= 5:
        angle_penalty = abs(o[4])  # absolute angle
    
    # --- Signal 4: Angular velocity penalty (from obs index 5) ---
    # Penalizes spinning
    angular_vel_penalty = 0.0
    if len(o) >= 6:
        angular_vel_penalty = abs(o[5])  # absolute angular velocity
    
    # --- Signal 5: Action cost ---
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 1.0
    elif action in [1, 3]:  # Side engines
        action_cost = 0.5
    
    # --- Signal 6: Contact bonus (from obs indices 6,7) ---
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = o[6] + o[7]  # Sum of ground contact indicators
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on movement and exploration, low penalties
    # Middle stage: balance movement with stability
    # Late stage: prioritize smooth landing and contact
    
    if training_progress < 0.3:
        # Early training: encourage movement, low penalties
        w_movement = 1.0
        w_vel = -0.1
        w_angle = -0.1
        w_angvel = -0.05
        w_action = -0.1
        w_contact = 0.0
    elif training_progress < 0.7:
        # Middle training: balance movement with stability
        w_movement = 0.5
        w_vel = -0.3
        w_angle = -0.3
        w_angvel = -0.2
        w_action = -0.2
        w_contact = 0.3
    else:
        # Late training: prioritize smooth landing and contact
        w_movement = 0.2
        w_vel = -0.5
        w_angle = -0.5
        w_angvel = -0.3
        w_action = -0.3
        w_contact = 0.8
    
    # Combine components
    reward = (w_movement * movement_toward_origin +
              w_vel * vel_penalty +
              w_angle * angle_penalty +
              w_angvel * angular_vel_penalty +
              w_action * action_cost +
              w_contact * contact_bonus)
    
    return reward