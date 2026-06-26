def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement toward origin - encourages reducing absolute values
    movement_toward_origin = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness - penalizes large changes (encourages stable control)
    transition_magnitude = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost - penalizes engine usage
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 1.0
    elif action in [1, 3]:  # Side engines
        action_cost = 0.5
    
    # Feature 4: Contact bonus - from info or observation
    # Using last two elements of obs (ground contact indicators)
    contact_bonus = 0.0
    if len(o) >= 2:
        contact_bonus = o[-1] + o[-2]  # Sum of ground contact indicators
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance movement and stability
    # Late stage: prioritize smooth control and contact
    
    if training_progress < 0.3:
        # Early training: encourage movement toward origin
        w_movement = 1.0
        w_smooth = -0.1
        w_action = -0.2
        w_contact = 0.0
    elif training_progress < 0.7:
        # Middle training: balance movement with stability
        w_movement = 0.7
        w_smooth = -0.3
        w_action = -0.3
        w_contact = 0.2
    else:
        # Late training: prioritize smooth landing and contact
        w_movement = 0.3
        w_smooth = -0.5
        w_action = -0.4
        w_contact = 0.5
    
    # Combine components
    reward = (w_movement * movement_toward_origin +
              w_smooth * transition_magnitude +
              w_action * action_cost +
              w_contact * contact_bonus)
    
    return reward