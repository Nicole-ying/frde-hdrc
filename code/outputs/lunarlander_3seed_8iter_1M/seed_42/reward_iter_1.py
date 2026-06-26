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

    # Feature 5: Additional dense signal - change in absolute values (signed)
    # This encourages moving towards zero (center) more directly
    signed_change = sum(n[i] - o[i] for i in range(len(o)))
    # Penalize moving away from zero (positive change when already positive, etc.)
    away_penalty = sum(max(0, o[i] * (n[i] - o[i])) for i in range(len(o)))

    # Feature 6: Velocity-like signal from consecutive observations
    # If we have info with previous obs, use it; otherwise approximate
    # For now, use the change as a proxy for velocity
    velocity_magnitude = sum((n[i] - o[i]) ** 2 for i in range(len(o))) ** 0.5

    # Combine features with stage weights
    # Increase the weight of abs_diff to make it more influential
    # Add a small bonus for moving towards center (signed_change * sign)
    center_bonus = -sum(o[i] * (n[i] - o[i]) for i in range(len(o)))
    center_bonus = max(0, center_bonus)  # Only positive when moving towards center

    reward = (
        early_weight * abs_diff * 0.2  # Increased from 0.1 to 0.2
        - late_weight * sq_change * 0.01
        - action_cost
        + early_weight * center_bonus * 0.05  # New: bonus for moving towards center
        - away_penalty * 0.02  # New: penalty for moving away from center
    )

    return reward