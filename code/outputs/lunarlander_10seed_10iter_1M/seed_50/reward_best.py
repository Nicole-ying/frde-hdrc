def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward center - encourage reducing absolute position
    # Use generic dimension-agnostic feature: sum of absolute differences
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(2))  # First 2 dims are position
    move_reward = pos_change * 0.5
    
    # Component 2: Velocity damping - encourage reducing velocity magnitude
    vel_change = sum(o[i]**2 - n[i]**2 for i in range(2, 4))  # Dims 2-3 are velocity
    vel_reward = vel_change * 0.3
    
    # Component 3: Stability - encourage upright orientation and low angular velocity
    angle_change = abs(o[4]) - abs(n[4])  # Dim 4 is angle
    angvel_change = abs(o[5]) - abs(n[5])  # Dim 5 is angular velocity
    stability_reward = (angle_change * 0.2 + angvel_change * 0.1)
    
    # Component 4: Ground contact bonus - encourage landing
    ground_contact = sum(n[6:8])  # Dims 6-7 are ground contact flags
    contact_reward = ground_contact * 0.5
    
    # Component 5: Action penalty - small cost for using engines
    action_cost = -0.01 * (1.0 if action != 0 else 0.0)
    
    # Stage weights based on training progress
    # Early: focus on movement and velocity
    # Middle: add stability
    # Late: emphasize ground contact
    if training_progress < 0.3:
        w_move = 1.0
        w_vel = 1.0
        w_stability = 0.3
        w_contact = 0.1
        w_action = 0.5
    elif training_progress < 0.7:
        w_move = 0.7
        w_vel = 0.7
        w_stability = 0.8
        w_contact = 0.5
        w_action = 0.3
    else:
        w_move = 0.3
        w_vel = 0.3
        w_stability = 0.5
        w_contact = 1.0
        w_action = 0.1
    
    # Combine components
    reward = (w_move * move_reward + 
              w_vel * vel_reward + 
              w_stability * stability_reward + 
              w_contact * contact_reward + 
              w_action * action_cost)
    
    return reward