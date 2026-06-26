def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Dimension-agnostic transition features
    # Feature 1: Change in absolute values (encourages moving towards center)
    # This was the most stable component across all iterations
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

    # Feature 6: Distance to zero (penalize being far from center)
    dist_to_center = sum(abs(o[i]) for i in range(len(o)))

    # Feature 7: Velocity magnitude (dense signal about movement)
    velocity_mag = sum((n[i] - o[i]) ** 2 for i in range(len(o))) ** 0.5

    # Combine features with stage weights
    # Based on historical analysis:
    # - abs_diff was the most stable component (best score when weight=0.1 in iter 0 and iter 3)
    # - center_progress helped slightly (iter 3 score -136 vs iter 0 score -136, similar)
    # - away_penalty hurt (score dropped from -136 to -155 when added in iter 1)
    # - velocity_mag didn't help (score dropped to -162 when added in iter 2)
    # - sq_change penalty seems neutral but safe
    # - action_cost is small and safe
    # - dist_to_center penalty is new in iter 3, seems neutral
    # Strategy: keep abs_diff at 0.1 (best weight), keep center_progress at 0.01 (safe),
    # remove dist_to_center penalty (unnecessary complexity), keep sq_change and action_cost
    # Add a small bonus for velocity magnitude only in early training to encourage exploration
    reward = (
        early_weight * abs_diff * 0.1  # Best weight from iter 0 and iter 3
        - late_weight * sq_change * 0.01  # Keep smoothness penalty
        - action_cost  # Small action penalty
        + early_weight * center_progress * 0.01  # Keep from iter 3
        + early_weight * velocity_mag * 0.001  # Very small exploration bonus
    )

    return reward