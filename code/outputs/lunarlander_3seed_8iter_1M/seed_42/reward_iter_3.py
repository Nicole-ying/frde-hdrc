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
    early_weight = 1.0 - training_progress
    late_weight = training_progress

    # Feature 5: Signed change towards center (positive when moving towards zero)
    center_progress = -sum(o[i] * (n[i] - o[i]) for i in range(len(o)))
    center_progress = max(0, center_progress)  # Only positive when moving towards center

    # Feature 6: Velocity magnitude (dense signal about movement)
    velocity_mag = sum((n[i] - o[i]) ** 2 for i in range(len(o))) ** 0.5

    # Feature 7: Distance to zero (penalize being far from center)
    dist_to_center = sum(abs(o[i]) for i in range(len(o)))

    # Combine features with stage weights
    # Based on historical analysis:
    # - abs_diff was the most stable component (present in all iterations, best score when weight=0.1)
    # - center_progress helped slightly in iter 1 (score -155 vs -136) but was too aggressive
    # - away_penalty hurt (score dropped from -136 to -155 when added)
    # - velocity_mag in iter 2 didn't help (score dropped further to -162)
    # - sq_change penalty seems neutral but safe
    # - action_cost is small and safe
    # Strategy: simplify, keep only the most reliable components, reduce complexity
    reward = (
        early_weight * abs_diff * 0.1  # Best weight from iter 0
        - late_weight * sq_change * 0.01  # Keep smoothness penalty
        - action_cost  # Small action penalty
        + early_weight * center_progress * 0.01  # Reduced from 0.02 to 0.01
        - dist_to_center * 0.001  # Small penalty for being far from center
    )

    return reward