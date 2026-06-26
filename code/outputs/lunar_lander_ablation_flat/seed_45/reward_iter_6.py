def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract state components from observation
    o = obs
    n = next_obs

    # Position components (indices 0,1)
    pos_x = o[0]
    pos_y = o[1]
    next_pos_x = n[0]
    next_pos_y = n[1]

    # Velocity components (indices 2,3)
    vel_x = o[2]
    vel_y = o[3]
    next_vel_x = n[2]
    next_vel_y = n[3]

    # Angle and angular velocity (indices 4,5)
    angle = o[4]
    ang_vel = o[5]
    next_angle = n[4]
    next_ang_vel = n[5]

    # Ground contact (indices 6,7)
    leg1_contact = o[6]
    leg2_contact = o[7]
    next_leg1_contact = n[6]
    next_leg2_contact = n[7]

    # --- Transition progress signals (directional) ---
    # Position progress toward center (x=0, y=0)
    pos_progress = (abs(pos_x) - abs(next_pos_x)) + (abs(pos_y) - abs(next_pos_y))

    # Velocity reduction (especially vertical)
    vel_progress = (abs(vel_y) - abs(next_vel_y)) + 0.3 * (abs(vel_x) - abs(next_vel_x))

    # Angle improvement toward upright
    angle_progress = abs(angle) - abs(next_angle)

    # Angular velocity reduction
    ang_vel_progress = abs(ang_vel) - abs(next_ang_vel)

    # --- State penalties (current state quality) ---
    # Penalty for being far from center
    distance_penalty = -0.02 * (abs(next_pos_x) + abs(next_pos_y))

    # Penalty for high vertical speed
    vertical_speed_penalty = -0.1 * abs(next_vel_y)

    # Penalty for being tilted
    angle_penalty = -0.05 * abs(next_angle)

    # Penalty for spinning
    ang_vel_penalty = -0.02 * abs(next_ang_vel)

    # --- Landing rewards ---
    # Ground contact reward (both legs)
    ground_contact_reward = 0.3 * (next_leg1_contact + next_leg2_contact)

    # Landing bonus: near center, low speed, both legs down
    landing_bonus = 0.0
    if abs(next_pos_x) < 0.15 and abs(next_pos_y) < 0.15 and abs(next_vel_y) < 0.1 and next_leg1_contact > 0.5 and next_leg2_contact > 0.5:
        landing_bonus = 2.0

    # --- Action penalty ---
    # Penalize main engine heavily, side engines lightly
    if action == 2:
        action_penalty = -0.15
    elif action in [1, 3]:
        action_penalty = -0.05
    else:
        action_penalty = 0.0

    # --- Smoothness penalty (prevent jerky behavior) ---
    smoothness_penalty = -0.005 * ((next_pos_x - pos_x)**2 + (next_pos_y - pos_y)**2 + (next_vel_x - vel_x)**2 + (next_vel_y - vel_y)**2)

    # --- Survival bonus (encourage longer episodes) ---
    survival_bonus = 0.02

    # Combine all components with fixed coefficients
    reward = (
        1.5 * pos_progress +
        2.0 * vel_progress +
        1.0 * angle_progress +
        0.5 * ang_vel_progress +
        distance_penalty +
        vertical_speed_penalty +
        angle_penalty +
        ang_vel_penalty +
        ground_contact_reward +
        landing_bonus +
        action_penalty +
        smoothness_penalty +
        survival_bonus
    )

    return reward