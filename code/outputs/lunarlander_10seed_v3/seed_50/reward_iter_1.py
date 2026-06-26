def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs

    # Transition progress: directional movement toward desirable states
    # Use change in absolute values as a generic directional signal
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Smoothness: penalize large state changes to encourage stable transitions
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # Action cost: penalize engine use to encourage efficiency
    action_cost = 0.0
    if action == 2:
        action_cost = 0.5
    elif action in [1, 3]:
        action_cost = 0.2

    # Leg contact bonus: reward ground contact (indices 6 and 7)
    leg_contact = o[6] + o[7]

    # Velocity penalty: discourage high speeds (indices 2 and 3 are velocities)
    vel_penalty = abs(o[2]) + abs(o[3])

    # Angle penalty: discourage tilting (index 4 is angle)
    angle_penalty = abs(o[4])

    # Stage-based weights
    w1 = 1.0 - 0.5 * training_progress
    w2 = 0.1 + 0.3 * training_progress
    w3 = 0.2 * (1.0 - training_progress)
    w4 = 0.5 * training_progress
    w5 = 0.3 * (1.0 - training_progress)
    w6 = 0.4 * (1.0 - training_progress)

    reward = (w1 * abs_change
              - w2 * squared_change
              - w3 * action_cost
              + w4 * leg_contact
              - w5 * vel_penalty
              - w6 * angle_penalty)

    return reward