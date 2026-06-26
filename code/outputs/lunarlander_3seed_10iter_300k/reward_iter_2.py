def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Dimension-agnostic transition features
    
    # Smoothness bonus - reward smooth, controlled movements
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Ground contact bonus from info (if available)
    ground_contact_bonus = 0.0
    if 'ground_contact' in info:
        ground_contact_bonus = 1.0 if info['ground_contact'] else 0.0
    
    # Episode length bonus - reward staying alive
    # info may contain step count or we can infer from training_progress
    survival_bonus = 0.01  # Small constant reward per step
    
    # Stage weights based on training progress
    # Early: focus on exploration and smoothness
    # Late: focus on ground contact and stability
    smoothness_weight = 1.0 - 0.5 * training_progress
    contact_weight = 0.5 * training_progress
    survival_weight = 0.1 * (1.0 - 0.5 * training_progress)
    
    # Combine components
    reward = (
        smoothness_weight * smoothness
        + contact_weight * ground_contact_bonus
        + survival_weight * survival_bonus
    )
    
    return float(reward)