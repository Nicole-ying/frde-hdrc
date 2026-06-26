def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large changes in absolute values
    # This encourages smooth transitions and stability
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_quality = abs_diff / len(o)
    
    # Component 2: Smoothness - penalize large squared changes
    # This encourages gentle, controlled movements
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness = -squared_diff / len(o)
    
    # Component 3: Action penalty - discourage excessive action usage
    # Convert action to a small cost
    action_cost = -0.01 * float(action)
    
    # Component 4: Progress-based exploration bonus
    # Early training: encourage exploration with positive movement
    # Late training: focus on stability and precision
    exploration_bonus = 0.0
    if training_progress < 0.3:
        exploration_bonus = 0.1 * abs(movement_quality)
    elif training_progress > 0.7:
        exploration_bonus = -0.05 * abs(movement_quality)
    
    # Stage weights that evolve with training progress
    w1 = 0.3 + 0.4 * training_progress  # movement quality weight increases
    w2 = 0.4 - 0.2 * training_progress  # smoothness weight decreases slightly
    w3 = 0.1  # constant small action penalty
    w4 = 0.2 * (1.0 - training_progress)  # exploration bonus decreases
    
    # Combine components
    reward = (w1 * movement_quality + 
              w2 * smoothness + 
              w3 * action_cost + 
              w4 * exploration_bonus)
    
    return reward