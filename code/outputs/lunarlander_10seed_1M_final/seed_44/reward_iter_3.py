def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Generic arrays for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Component 1: Convergence reward - reward when observations move toward zero
    # Sum of (|current| - |next|) - positive when values decrease in magnitude
    convergence = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    convergence_reward = convergence * 0.5
    
    # Component 2: Smoothness penalty - penalize large squared changes
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.2
    
    # Component 3: Action penalty - small cost for taking actions to encourage efficiency
    action_cost = -0.01 * float(action)
    
    # Component 4: Survival bonus - reward staying alive
    survival_bonus = 0.1
    
    # Component 5: Velocity stability - penalize large absolute velocity changes
    # Using first 4 obs dimensions (pos x, pos y, vel x, vel y) to detect movement
    vel_change = sum(abs(n[i] - o[i]) for i in range(min(4, len(o))))
    velocity_penalty = -vel_change * 0.05
    
    # Stage weights based on training_progress
    # Early training: focus on exploration, low penalties
    # Mid training: balance all components
    # Late training: emphasize convergence and smoothness
    if training_progress < 0.3:
        # Early stage - prioritize survival and basic convergence
        w_convergence = 0.3
        w_smoothness = 0.1
        w_action = 0.1
        w_survival = 1.0
        w_velocity = 0.05
    elif training_progress < 0.7:
        # Mid stage - balance all components
        w_convergence = 1.0
        w_smoothness = 0.5
        w_action = 0.3
        w_survival = 0.5
        w_velocity = 0.2
    else:
        # Late stage - emphasize convergence and smoothness
        w_convergence = 1.5
        w_smoothness = 1.0
        w_action = 0.5
        w_survival = 0.2
        w_velocity = 0.3
    
    # Combine components with stage weights
    reward = (w_convergence * convergence_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost +
              w_survival * survival_bonus +
              w_velocity * velocity_penalty)
    
    return reward