def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs

    # Transition progress: reward for moving toward zero in position and velocity
    pos_x_progress = abs(o[0]) - abs(n[0])
    pos_y_progress = abs(o[1]) - abs(n[1])
    vel_x_progress = abs(o[2]) - abs(n[2])
    vel_y_progress = abs(o[3]) - abs(n[3])
    angle_progress = abs(o[4]) - abs(n[4])
    ang_vel_progress = abs(o[5]) - abs(n[5])

    # Penalty for large absolute values (encourages staying near zero)
    pos_x_penalty = abs(n[0])
    pos_y_penalty = abs(n[1])
    vel_x_penalty = abs(n[2])
    vel_y_penalty = abs(n[3])
    angle_penalty = abs(n[4])
    ang_vel_penalty = abs(n[5])

    # Ground contact bonus
    ground_contact_bonus = n[6] + n[7]

    # Action penalty for fuel efficiency
    action_penalty = 0.01 * (action == 2) + 0.005 * (action in [1, 3])

    # Combine: progress signals encourage improvement, penalties discourage bad states
    reward = (
        1.0 * pos_x_progress
        + 0.5 * pos_y_progress
        + 0.3 * vel_x_progress
        + 0.5 * vel_y_progress
        + 0.2 * angle_progress
        + 0.05 * ang_vel_progress
        - 0.3 * pos_x_penalty
        - 0.3 * pos_y_penalty
        - 0.2 * vel_x_penalty
        - 0.3 * vel_y_penalty
        - 0.1 * angle_penalty
        - 0.05 * ang_vel_penalty
        + 1.0 * ground_contact_bonus
        - action_penalty
    )

    return reward