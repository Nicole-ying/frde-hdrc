def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Dimension-agnostic transition features
    
    # 1. Movement toward center (encourage staying near origin)
    # Use sum of absolute values as a proxy for distance from origin
    dist_old = sum(abs(o[i]) for i in range(len(o)))
    dist_new = sum(abs(n[i]) for i in range(len(n)))
    movement_reward = (dist_old - dist_new) * 0.1
    
    # 2. Smoothness penalty - penalize large changes between steps
    change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -change * 0.01
    
    # 3. Action penalty - discourage excessive action usage
    action_cost = -0.01 * float(action)
    
    # 4. Contact bonus - if contact sensors change, that's interesting
    # Last two dimensions are contact sensors (indices 6 and 7)
    contact_bonus = 0.0
    if len(o) >= 8:
        old_contacts = o[6] + o[7]
        new_contacts = n[6] + n[7]
        contact_bonus = (new_contacts - old_contacts) * 0.5
    
    # Combine with stage weights based on training_progress
    # Early training: focus on movement and exploration
    # Late training: focus on smoothness and precision
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    reward = (
        early_weight * (movement_reward + action_cost) +
        late_weight * (smoothness_penalty + contact_bonus)
    )
    
    return reward