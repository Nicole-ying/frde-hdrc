def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Generic arrays for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Component 1: Movement magnitude change (encourages reducing absolute values)
    movement_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness penalty (discourages large changes)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Component 3: Action cost (small penalty for any action)
    action_cost = -0.01 * (1.0 if action != 0 else 0.0)
    
    # Stage-based weights that evolve with training_progress
    # Early training: focus on movement reduction
    # Late training: focus on smoothness and stability
    w1 = 1.0 - 0.5 * training_progress  # movement weight decreases
    w2 = 0.5 * training_progress        # smoothness weight increases
    w3 = 0.1                            # constant small action penalty
    
    # Combine components
    reward = w1 * movement_change + w2 * smoothness + w3 * action_cost
    
    return reward