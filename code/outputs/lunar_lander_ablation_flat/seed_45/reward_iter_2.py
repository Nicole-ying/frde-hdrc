def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state components
    o = obs
    n = next_obs

    # Transition progress: reward movement toward center (directional)
    pos_progress = abs(o[0]) - abs(n[0]) + abs(o[1]) - abs(n[1])

    # Velocity reduction: reward slowing down (directional)
    vel_progress = abs(o[2]) - abs(n[2]) + abs(o[3]) - abs(n[3])

    # Angular stability: reward upright orientation (directional)
    angle_progress = abs(o[4]) - abs(n[4])

    # Angular velocity reduction: reward less spinning (directional)
    ang_vel_progress = abs(o[5]) - abs(n[5])

    # Ground contact bonus: reward both legs touching
    leg_contact = n[6] + n[7]

    # Action penalty: discourage unnecessary engine use
    # action 2 = main engine, action 1 or 3 = side engines
    action_penalty = 0.0
    if action == 2:
        action_penalty = -0.2
    elif action == 1 or action == 3:
        action_penalty = -0.1

    # Smoothness penalty: penalize large changes in state (undirected, but useful for stability)
    smoothness = -0.005 * ((n[0] - o[0])**2 + (n[1] - o[1])**2 + (n[2] - o[2])**2 + (n[3] - o[3])**2)

    # Survival bonus: reward staying alive (longer episodes)
    survival = 0.1

    # Combine all signals with fixed coefficients
    reward = (
        1.5 * pos_progress +
        2.0 * vel_progress +
        1.0 * angle_progress +
        0.5 * ang_vel_progress +
        2.0 * leg_contact +
        action_penalty +
        smoothness +
        survival
    )

    return reward