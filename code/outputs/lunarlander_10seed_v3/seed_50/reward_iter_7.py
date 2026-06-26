def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude: encourage moving towards target (sum of absolute changes)
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 2. Smoothness: penalize large changes (encourages stable control)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 3. Action cost: penalize engine usage
    action_cost = -0.01 * float(action)
    
    # Stage-based weights that evolve with training
    # Early training: focus on movement and exploration
    # Late training: focus on smoothness and precision
    w_movement = 1.0 - 0.5 * training_progress
    w_smoothness = 0.5 * training_progress
    w_action = 0.1 * (1.0 - 0.5 * training_progress)
    
    # Combine components
    reward = w_movement * movement + w_smoothness * smoothness + w_action * action_cost
    
    return reward