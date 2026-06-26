def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs

    # Movement toward center (directional: positive when moving toward zero)
    move_x = abs(o[0]) - abs(n[0])
    move_y = abs(o[1]) - abs(n[1])
    movement_reward = move_x * 0.5 + move_y * 0.5

    # Penalize high velocity (squared for smooth gradient)
    vel_penalty = (n[2] ** 2 + n[3] ** 2) * 0.2

    # Penalize angle deviation and angular velocity
    angle_penalty = abs(n[4]) * 0.8 + abs(n[5]) * 0.3

    # Reward ground contact (both legs)
    contact_bonus = (n[6] + n[7]) * 1.0

    # Penalize action usage (main engine costlier)
    if action == 2:
        action_cost = 0.02
    elif action in [1, 3]:
        action_cost = 0.01
    else:
        action_cost = 0.0

    # Penalize sudden changes in velocity and angle (smoothness)
    vel_change = ((n[2] - o[2]) ** 2 + (n[3] - o[3]) ** 2) * 0.1
    angle_change = (n[4] - o[4]) ** 2 * 0.15
    smoothness_penalty = vel_change + angle_change

    # Reward being close to center (position bonus)
    pos_bonus = (1.0 - abs(n[0])) * 0.2 + (1.0 - abs(n[1])) * 0.2

    # Combine all components
    reward = (
        movement_reward
        - vel_penalty
        - angle_penalty
        + contact_bonus
        - action_cost
        - smoothness_penalty
        + pos_bonus
    )

    return reward