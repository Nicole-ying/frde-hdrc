def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement magnitude - squared difference between consecutive observations
    diff_sq = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 2: Absolute value change - encourages moving toward smaller absolute values
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 3: Action penalty - discourage excessive action usage
    action_penalty = 0.0
    if action == 2:  # Main engine
        action_penalty = -0.1
    elif action in [1, 3]:  # Side engines
        action_penalty = -0.05
    
    # Feature 4: Stability - penalize large changes in observation (smoothness)
    stability = -sum(abs(n[i] - o[i]) for i in range(len(o))) * 0.01
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on exploration and movement
    # Mid training: balance movement and stability
    # Late training: focus on precision and stability
    
    if training_progress < 0.3:
        # Early stage: encourage exploration through movement
        w_diff = 0.5
        w_abs = 0.3
        w_action = 0.1
        w_stability = 0.1
    elif training_progress < 0.7:
        # Middle stage: balance movement with stability
        w_diff = 0.3
        w_abs = 0.3
        w_action = 0.2
        w_stability = 0.2
    else:
        # Late stage: focus on precision and stable behavior
        w_diff = 0.2
        w_abs = 0.4
        w_action = 0.2
        w_stability = 0.2
    
    # Combine components
    reward = (
        w_diff * diff_sq +
        w_abs * abs_change +
        w_action * action_penalty +
        w_stability * stability
    )
    
    # Add small constant to encourage exploration early
    if training_progress < 0.2:
        reward += 0.01
    
    return float(reward)