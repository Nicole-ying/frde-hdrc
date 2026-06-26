def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs

    # Extract all state components
    pos_x_o, pos_y_o = o[0], o[1]
    vel_x_o, vel_y_o = o[2], o[3]
    angle_o = o[4]
    ang_vel_o = o[5]
    leg1_o, leg2_o = o[6], o[7]

    pos_x_n, pos_y_n = n[0], n[1]
    vel_x_n, vel_y_n = n[2], n[3]
    angle_n = n[4]
    ang_vel_n = n[5]
    leg1_n, leg2_n = n[6], n[7]

    # Directional progress toward center (x=0)
    center_progress = abs(pos_x_o) - abs(pos_x_n)

    # Directional progress toward ground (y=0) - landing pad height
    ground_progress = abs(pos_y_o) - abs(pos_y_n)

    # Velocity reduction - directional
    vel_mag_o = (vel_x_o ** 2 + vel_y_o ** 2) ** 0.5
    vel_mag_n = (vel_x_n ** 2 + vel_y_n ** 2) ** 0.5
    vel_reduction = vel_mag_o - vel_mag_n

    # Angle stabilization - directional
    angle_progress = abs(angle_o) - abs(angle_n)

    # Angular velocity damping - directional
    ang_vel_progress = abs(ang_vel_o) - abs(ang_vel_n)

    # Leg contact reward (both legs = successful landing)
    leg_contact = leg1_n + leg2_n

    # Action penalty (encourage fuel efficiency)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.2
    elif action == 1 or action == 3:  # Side engines
        action_cost = 0.1

    # Survival bonus for staying alive
    survival = 0.1

    # Combine with fixed coefficients - keep per-step reward in reasonable range
    reward = (
        center_progress * 5.0 +
        ground_progress * 3.0 +
        vel_reduction * 3.0 +
        angle_progress * 2.0 +
        ang_vel_progress * 1.0 +
        leg_contact * 3.0 +
        survival -
        action_cost
    )

    return reward