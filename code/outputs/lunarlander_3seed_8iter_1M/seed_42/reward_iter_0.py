def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Dimension-agnostic transition features
    # Feature 1: Change in absolute values (encourages moving towards center)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (penalizes large changes, encourages smooth movement)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (discourage unnecessary actions)
    action_cost = 0.01 * action
    
    # Feature 4: Progress-based weighting
    # Early training: focus on exploration (abs_diff)
    # Late training: focus on precision (sq_change)
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Combine features with stage weights
    reward = (
        early_weight * abs_diff * 0.1  # Encourage moving towards center early
        - late_weight * sq_change * 0.01  # Penalize large changes late
        - action_cost  # Small action penalty
    )
    
    return reward