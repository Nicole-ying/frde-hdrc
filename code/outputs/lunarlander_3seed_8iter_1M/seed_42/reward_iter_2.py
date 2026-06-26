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

    # Feature 5: Velocity magnitude (dense signal about movement)
    velocity_mag = sum((n[i] - o[i]) ** 2 for i in range(len(o))) ** 0.5

    # Feature 6: Signed change towards center (positive when moving towards zero)
    center_progress = -sum(o[i] * (n[i] - o[i]) for i in range(len(o)))
    center_progress = max(0, center_progress)  # Only positive when moving towards center

    # Combine features with stage weights
    # Keep abs_diff as main signal (was in both iterations, score improved when weight increased)
    # Remove away_penalty (added in iter 1, score got worse)
    # Keep center_progress but reduce its weight (was too aggressive in iter 1)
    # Add velocity magnitude as a dense exploration bonus (new signal)
    reward = (
        early_weight * abs_diff * 0.15  # Moderate weight for moving towards center
        - late_weight * sq_change * 0.01  # Penalize large changes late
        - action_cost  # Small action penalty
        + early_weight * center_progress * 0.02  # Small bonus for moving towards center
        + early_weight * velocity_mag * 0.005  # Small exploration bonus for movement
    )

    return reward