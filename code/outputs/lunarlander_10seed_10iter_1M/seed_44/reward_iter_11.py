def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # --- Core transition signals (all dimension-agnostic) ---
    
    # 1. Directional improvement: reward moving toward zero (desirable state)
    # This is the ONLY component that consistently correlated with better scores
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 2. Movement magnitude - for exploration, but penalized to avoid chaos
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 3. Action cost - penalize engine usage (small, to avoid early death)
    action_cost = 0.0
    if action == 2:  # main engine
        action_cost = 0.2
    elif action in [1, 3]:  # side engines
        action_cost = 0.1
    
    # 4. Survival bonus - reward staying alive (critical for longer episodes)
    survival_bonus = 0.1
    
    # --- Stage-based weights ---
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # --- Component rewards with proper scaling ---
    # Directional improvement: primary signal throughout (proven effective in iter 5)
    abs_change_reward = abs_change * 1.0
    
    # Movement penalty: discourage chaotic movement (was harmful as reward)
    movement_penalty = -movement * 0.05
    
    # Action penalty
    action_penalty = -action_cost
    
    # --- Combine with stage weights ---
    # Simplified structure: focus on what worked (abs_change + survival)
    # Remove complex stability/velocity/contact signals that didn't help
    reward = (
        early_weight * (
            abs_change_reward * 0.6 + 
            survival_bonus * 0.3 + 
            movement_penalty * 0.1
        ) +
        mid_weight * (
            abs_change_reward * 0.7 + 
            survival_bonus * 0.2 + 
            movement_penalty * 0.1
        ) +
        late_weight * (
            abs_change_reward * 0.8 + 
            survival_bonus * 0.1 + 
            movement_penalty * 0.1
        ) +
        action_penalty
    )
    
    return reward