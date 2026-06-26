def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude: squared difference between consecutive observations
    diff_sq = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 2. Absolute change: sum of absolute differences
    abs_change = sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # 3. Convergence signal: if absolute values are decreasing (moving toward zero)
    convergence = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 4. Action penalty (small cost for taking actions)
    action_cost = 0.01 * abs(action)
    
    # Stage weights based on training progress
    # Early stage: focus on exploration (movement)
    # Middle stage: balance movement and convergence
    # Late stage: focus on convergence (stabilization)
    if training_progress < 0.3:
        # Early exploration
        w_diff = 1.0
        w_change = 0.5
        w_converge = 0.2
        w_action = 0.1
    elif training_progress < 0.7:
        # Middle stage - balanced
        w_diff = 0.6
        w_change = 0.3
        w_converge = 0.8
        w_action = 0.05
    else:
        # Late stage - convergence focused
        w_diff = 0.2
        w_change = 0.1
        w_converge = 1.5
        w_action = 0.02
    
    # Combine components
    reward = (
        w_diff * diff_sq +
        w_change * abs_change +
        w_converge * convergence -
        w_action * action_cost
    )
    
    return reward