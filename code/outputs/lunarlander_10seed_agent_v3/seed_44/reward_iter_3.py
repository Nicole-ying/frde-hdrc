def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude - encourage movement toward target
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 2. Smoothness - penalize large changes
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 3. Action penalty - discourage excessive action
    action_cost = -0.01 * abs(action - 1.5)  # Center action around 1.5 for discrete 0-3
    
    # Stage-based weights
    if training_progress < 0.3:
        # Early stage: focus on movement and exploration
        w_movement = 1.0
        w_smoothness = 0.1
        w_action = 0.01
    elif training_progress < 0.7:
        # Middle stage: balance movement with smoothness
        w_movement = 0.8
        w_smoothness = 0.5
        w_action = 0.05
    else:
        # Late stage: refine control
        w_movement = 0.5
        w_smoothness = 1.0
        w_action = 0.1
    
    # Combine components
    reward = (w_movement * movement + 
              w_smoothness * smoothness + 
              w_action * action_cost)
    
    return reward