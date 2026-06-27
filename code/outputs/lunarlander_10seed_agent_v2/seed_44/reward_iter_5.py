def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Generic transition signals
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability (reducing absolute values)
    movement_towards_zero = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness of transition (penalize large changes)
    transition_smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Component 3: Action cost (penalize taking actions)
    action_cost = -0.01 * abs(action - 1.5)  # Center action around 1.5 for Discrete(4)
    
    # Stage-based weights that evolve with training
    # Early stage: focus on movement towards stability
    # Middle stage: balance stability and smoothness
    # Late stage: prioritize smoothness and fine control
    if training_progress < 0.3:
        w1 = 1.0  # movement weight
        w2 = 0.1  # smoothness weight
        w3 = 0.5  # action cost weight
    elif training_progress < 0.7:
        w1 = 0.6
        w2 = 0.4
        w3 = 0.3
    else:
        w1 = 0.3
        w2 = 0.7
        w3 = 0.2
    
    # Combine components
    reward = w1 * movement_towards_zero + w2 * transition_smoothness + w3 * action_cost
    
    return float(reward)