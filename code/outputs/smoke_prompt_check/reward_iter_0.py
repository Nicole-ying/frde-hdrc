def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs
    
    # Generic transition features
    # Movement magnitude (how much the state changed)
    state_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Absolute value reduction (encourages moving toward zero)
    abs_reduction = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Stage-based weights
    # Early training: focus on exploration and state changes
    # Late training: focus on convergence and stability
    early_weight = max(0.0, 1.0 - training_progress)
    late_weight = min(1.0, training_progress)
    
    # Reward components
    exploration_bonus = early_weight * 0.1 * state_change
    convergence_bonus = late_weight * 0.5 * abs_reduction
    action_penalty = -action_cost * (1.0 - 0.5 * training_progress)
    
    # Combine components
    reward = exploration_bonus + convergence_bonus + action_penalty
    
    return reward