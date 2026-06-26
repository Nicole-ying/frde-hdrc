def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state components
    o = obs
    n = next_obs

    # Position progress toward center (directional)
    pos_progress = abs(o[0]) - abs(n[0]) + abs(o[1]) - abs(n[1])

    # Velocity reduction (directional)
    vel_progress = abs(o[2]) - abs(n[2]) + abs(o[3]) - abs(n[3])

    # Angular stability (directional)
    angle_progress = abs(o[4]) - abs(n[4])

    # Angular velocity reduction (directional)
    ang_vel_progress = abs(o[5]) - abs(n[5])

    # Ground contact bonus
    leg_contact = n[6] + n[7]

    # Action penalty
    action_penalty = 0.0
    if action == 2:
        action_penalty = -0.15
    elif action == 1 or action == 3:
        action_penalty = -0.08

    # Smoothness penalty (undirected but helps stability)
    smoothness = -0.008 * ((n[0] - o[0])**2 + (n[1] - o[1])**2 + (n[2] - o[2])**2 + (n[3] - o[3])**2)

    # Survival bonus
    survival = 0.05

    # Combine all signals with fixed coefficients
    reward = (
        2.5 * pos_progress +
        2.5 * vel_progress +
        1.5 * angle_progress +
        1.0 * ang_vel_progress +
        1.5 * leg_contact +
        action_penalty +
        smoothness +
        survival
    )

    return reward