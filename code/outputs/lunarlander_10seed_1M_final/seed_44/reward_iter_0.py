def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Generic arrays for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large absolute changes
    # Sum of absolute differences between current and next observation components
    abs_diff = sum(abs(o[i] - n[i]) for i in range(len(o)))
    stability_reward = -abs_diff * 0.1
    
    # Component 2: Convergence reward - reward when observations move toward zero
    # Sum of (|current| - |next|) - positive when values decrease in magnitude
    convergence = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    convergence_reward = convergence * 0.5
    
    # Component 3: Smoothness reward - penalize large squared changes
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_reward = -squared_diff * 0.2
    
    # Component 4: Action penalty - small cost for taking actions to encourage efficiency
    action_cost = -0.01 * float(action)
    
    # Stage weights based on training_progress
    # Early training: focus on stability and convergence
    # Mid training: balance all components
    # Late training: emphasize convergence and smoothness
    if training_progress < 0.3:
        # Early stage - prioritize stability and basic convergence
        w_stability = 1.0
        w_convergence = 0.5
        w_smoothness = 0.3
        w_action = 0.1
    elif training_progress < 0.7:
        # Mid stage - balance all components
        w_stability = 0.5
        w_convergence = 1.0
        w_smoothness = 0.7
        w_action = 0.3
    else:
        # Late stage - emphasize convergence and smoothness
        w_stability = 0.2
        w_convergence = 1.5
        w_smoothness = 1.0
        w_action = 0.5
    
    # Combine components with stage weights
    reward = (w_stability * stability_reward + 
              w_convergence * convergence_reward + 
              w_smoothness * smoothness_reward + 
              w_action * action_cost)
    
    return reward