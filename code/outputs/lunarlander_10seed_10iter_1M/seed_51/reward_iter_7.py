def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # ---- Component 1: Directional stability (reward moving toward zero) ----
    # This was present in all iterations and correlated with longer episodes
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    stability_reward = abs_diff / (len(o) * 2.0)

    # ---- Component 2: Height reduction (encourage descent) ----
    # Height at index 1 - positive means above pad
    height = abs(o[1]) if len(o) > 1 else 0.0
    next_height = abs(n[1]) if len(n) > 1 else 0.0
    height_reduction = height - next_height
    height_reward = height_reduction * 2.0

    # ---- Component 3: Angle stability (discourage tipping) ----
    # Angle at index 4, angular velocity at index 5
    angle = abs(o[4]) if len(o) > 4 else 0.0
    next_angle = abs(n[4]) if len(n) > 4 else 0.0
    ang_vel = abs(o[5]) if len(o) > 5 else 0.0
    next_ang_vel = abs(n[5]) if len(n) > 5 else 0.0
    angle_reduction = (angle - next_angle) + (ang_vel - next_ang_vel) * 0.5
    angle_reward = angle_reduction * 1.5

    # ---- Component 4: Velocity control (discourage high speed) ----
    # Velocity at indices 2 and 3
    vel_x = abs(o[2]) if len(o) > 2 else 0.0
    vel_y = abs(o[3]) if len(o) > 3 else 0.0
    next_vel_x = abs(n[2]) if len(n) > 2 else 0.0
    next_vel_y = abs(n[3]) if len(n) > 3 else 0.0
    vel_reduction = (vel_x - next_vel_x) + (vel_y - next_vel_y)
    velocity_reward = vel_reduction * 1.0

    # ---- Component 5: Contact bonus (reward leg contact) ----
    leg_contact_bonus = 0.0
    if len(o) >= 8 and len(n) >= 8:
        prev_contact = o[6] + o[7]
        curr_contact = n[6] + n[7]
        if curr_contact > prev_contact:
            leg_contact_bonus = 2.0
        elif curr_contact > 0.5:
            leg_contact_bonus = 0.5

    # ---- Component 6: Smoothness penalty (discourage jerky motion) ----
    smoothness_penalty = 0.0
    if len(o) > 0:
        squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
        smoothness_penalty = -squared_diff / (len(o) * 40.0)

    # ---- Component 7: Action cost (discourage unnecessary engine use) ----
    if action == 2:
        action_cost = -0.3
    elif action in [1, 3]:
        action_cost = -0.15
    else:
        action_cost = 0.0

    # ---- Stage-based weights ----
    # Early: focus on stability and descent
    # Mid: balance all components
    # Late: emphasize landing precision
    if training_progress < 0.3:
        w_stability = 1.0
        w_height = 1.5
        w_angle = 1.0
        w_velocity = 1.0
        w_contact = 0.3
        w_smoothness = 0.5
        w_action = 0.3
    elif training_progress < 0.7:
        w_stability = 0.7
        w_height = 1.2
        w_angle = 1.5
        w_velocity = 0.8
        w_contact = 1.0
        w_smoothness = 0.7
        w_action = 0.5
    else:
        w_stability = 0.5
        w_height = 1.0
        w_angle = 2.0
        w_velocity = 0.5
        w_contact = 2.0
        w_smoothness = 0.3
        w_action = 0.6

    # ---- Combine components ----
    reward = (w_stability * stability_reward +
              w_height * height_reward +
              w_angle * angle_reward +
              w_velocity * velocity_reward +
              w_contact * leg_contact_bonus +
              w_smoothness * smoothness_penalty +
              w_action * action_cost)

    return reward