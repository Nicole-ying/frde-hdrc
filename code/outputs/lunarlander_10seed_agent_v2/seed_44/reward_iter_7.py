def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features - dimension agnostic
    # Feature 1: Movement magnitude (encourages exploration early, precision later)
    movement = sum(abs(o[i] - n[i]) for i in range(len(o)))
    
    # Feature 2: Absolute state change (captures any state transition)
    state_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Approach to stable states (penalize large absolute values)
    stability = -sum(abs(n[i]) for i in range(len(o)))
    
    # Feature 4: Action penalty (discourage excessive action usage)
    action_cost = -0.01 * abs(action)
    
    # Stage-based weights that evolve with training_progress
    # Early training: explore more, focus on movement and state changes
    # Late training: exploit stability and precision
    w_movement = 0.5 * (1.0 - training_progress) + 0.1 * training_progress
    w_state_change = 0.3 * (1.0 - training_progress) + 0.2 * training_progress
    w_stability = 0.1 * (1.0 - training_progress) + 0.5 * training_progress
    w_action = 0.1 * (1.0 - training_progress) + 0.2 * training_progress
    
    # Combine components
    reward = (w_movement * movement + 
              w_state_change * state_change + 
              w_stability * stability + 
              w_action * action_cost)
    
    return reward