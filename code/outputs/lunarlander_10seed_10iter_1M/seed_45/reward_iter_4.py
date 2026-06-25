def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Directional movement toward zero (stability signal)
    # Reward reducing absolute values of state components
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    stability_reward = abs_diff * 0.3
    
    # Component 2: Smoothness penalty - penalize large changes
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.05
    
    # Component 3: Action cost - penalize taking actions
    action_cost = -0.02 if action != 0 else 0.0
    
    # Component 4: Contact bonus from info if available
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = 0.3 if info['contact'] else 0.0
    
    # Component 5: Velocity penalty - penalize high velocity changes (crash prevention)
    vel_change = 0.0
    if len(o) >= 4:
        vel_change = abs(n[2] - o[2]) + abs(n[3] - o[3])
    velocity_penalty = -vel_change * 0.1
    
    # Component 6: Survival bonus - reward staying alive
    survival_bonus = 0.05
    
    # Component 7: Angular velocity penalty - prevent spinning
    angular_vel_penalty = 0.0
    if len(o) >= 6:
        angular_vel_penalty = -abs(n[5]) * 0.2
    
    # Stage-based weighting - simplified and balanced
    if training_progress < 0.3:
        # Early: focus on stability and survival
        w_stability = 1.5
        w_smoothness = 0.3
        w_action = 0.1
        w_contact = 0.0
        w_velocity = 0.5
        w_survival = 1.0
        w_angular = 0.5
    elif training_progress < 0.6:
        # Mid: balance all components
        w_stability = 1.0
        w_smoothness = 0.5
        w_action = 0.2
        w_contact = 0.5
        w_velocity = 0.3
        w_survival = 0.5
        w_angular = 0.3
    else:
        # Late: emphasize contact and stability
        w_stability = 0.8
        w_smoothness = 0.3
        w_action = 0.1
        w_contact = 1.5
        w_velocity = 0.2
        w_survival = 0.3
        w_angular = 0.2
    
    # Combine components
    reward = (w_stability * stability_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus +
              w_velocity * velocity_penalty +
              w_survival * survival_bonus +
              w_angular * angular_vel_penalty)
    
    return reward