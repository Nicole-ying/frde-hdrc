def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement magnitude - encourage reducing absolute values (approaching target)
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness - penalize large changes (encourage stable control)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty - small cost for taking actions (encourage efficiency)
    action_cost = -0.01 * abs(action - 1.5)  # Discrete action 0-3, center at 1.5
    
    # Feature 4: Leg contact bonus - encourage landing (from info if available)
    leg_contact_bonus = 0.0
    if 'leg_contact' in info:
        leg_contact_bonus = 1.0 if info['leg_contact'] else 0.0
    
    # Stage-based weighting
    # Early stage: focus on movement and exploration
    # Mid stage: balance movement and smoothness
    # Late stage: prioritize smoothness and leg contact
    if training_progress < 0.3:
        w_movement = 1.0
        w_smoothness = 0.1
        w_action = 0.5
        w_leg = 0.0
    elif training_progress < 0.7:
        w_movement = 0.7
        w_smoothness = 0.5
        w_action = 0.3
        w_leg = 0.5
    else:
        w_movement = 0.3
        w_smoothness = 1.0
        w_action = 0.1
        w_leg = 2.0
    
    # Combine components
    reward = (w_movement * movement + 
              w_smoothness * smoothness + 
              w_action * action_cost + 
              w_leg * leg_contact_bonus)
    
    return reward