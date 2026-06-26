def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward target (based on position changes)
    # Use first 4 dimensions which typically represent position and velocity
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Component 2: Smoothness - penalize large changes in state
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(8))
    
    # Component 3: Action cost - penalize engine usage
    action_cost = -0.01 * float(action)
    
    # Component 4: Ground contact bonus (dimensions 6 and 7 are contact indicators)
    ground_contact = sum(n[6:8]) if len(n) >= 8 else 0.0
    
    # Stage weights based on training progress
    if training_progress < 0.3:
        # Early training: focus on movement and exploration
        w1, w2, w3, w4 = 1.0, 0.1, 0.5, 0.0
    elif training_progress < 0.7:
        # Mid training: balance all components
        w1, w2, w3, w4 = 0.5, 0.3, 0.3, 0.5
    else:
        # Late training: emphasize smoothness and ground contact
        w1, w2, w3, w4 = 0.2, 0.5, 0.1, 1.0
    
    # Combine components
    reward = w1 * pos_change + w2 * smoothness + w3 * action_cost + w4 * ground_contact
    
    return reward