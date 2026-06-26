def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability - penalize large changes in absolute values
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_change * 0.1
    
    # Component 2: Smoothness - penalize large squared differences
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.05
    
    # Component 3: Action cost - small penalty for taking actions
    action_cost = -0.01 * float(action)
    
    # Component 4: Progress bonus - encourage consistent improvement
    progress_bonus = 0.0
    if training_progress > 0.5:
        progress_bonus = 0.02 * (training_progress - 0.5)
    
    # Stage weights based on training_progress
    stage_weight_1 = 1.0 - 0.5 * training_progress  # Movement weight decreases
    stage_weight_2 = 0.5 + 0.5 * training_progress   # Smoothness weight increases
    stage_weight_3 = 1.0                              # Action cost stays constant
    stage_weight_4 = 0.0 if training_progress < 0.3 else (training_progress - 0.3) / 0.7  # Progress bonus ramps up
    
    # Combine components
    reward = (stage_weight_1 * movement_reward + 
              stage_weight_2 * smoothness_penalty + 
              stage_weight_3 * action_cost + 
              stage_weight_4 * progress_bonus)
    
    return reward