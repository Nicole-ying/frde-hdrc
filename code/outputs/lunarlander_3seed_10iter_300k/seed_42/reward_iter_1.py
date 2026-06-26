def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as arrays
    o = obs
    n = next_obs

    # Generic transition signals
    # Movement magnitude (change in position/velocity)
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))

    # Smoothness penalty (acceleration/jerk)
    smoothness = sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # Action penalty (encourage efficient control)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0
    elif action == 1 or action == 3:
        action_cost = 0.1
    elif action == 2:
        action_cost = 0.2

    # Contact bonus (ground contact signals)
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = (n[6] + n[7]) * 0.5  # leg contact signals

    # Stage weights based on training progress
    early_weight = 1.0 - training_progress
    late_weight = training_progress

    # Combine components with stage weighting
    reward = (
        early_weight * (movement * 0.1 - smoothness * 0.01 - action_cost) +
        late_weight * (contact_bonus * 2.0 - smoothness * 0.005) +
        contact_bonus * 0.5  # Always reward contact
    )

    return reward