def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs

    # Position toward center (directional: reducing absolute position)
    pos_x_progress = abs(o[0]) - abs(n[0])
    pos_y_progress = abs(o[1]) - abs(n[1])
    position_reward = 0.5 * (pos_x_progress + pos_y_progress)

    # Velocity toward zero (directional: reducing absolute velocity)
    vel_x_progress = abs(o[2]) - abs(n[2])
    vel_y_progress = abs(o[3]) - abs(n[3])
    velocity_reward = 0.3 * (vel_x_progress + vel_y_progress)

    # Angle toward zero (directional: reducing absolute angle)
    angle_progress = abs(o[4]) - abs(n[4])
    angle_reward = 0.2 * angle_progress

    # Angular velocity toward zero (directional: reducing absolute angular velocity)
    angvel_progress = abs(o[5]) - abs(n[5])
    angvel_reward = 0.1 * angvel_progress

    # Ground contact bonus (both legs)
    contact_bonus = 0.5 * (n[6] + n[7])

    # Penalize large velocity changes (smoothness)
    vel_change_x = (n[2] - o[2]) ** 2
    vel_change_y = (n[3] - o[3]) ** 2
    smoothness_penalty = -0.05 * (vel_change_x + vel_change_y)

    # Action cost for main engine and side engines
    action_cost = 0.01 * (action == 2) + 0.005 * (action in [1, 3])

    reward = (position_reward + velocity_reward + angle_reward + 
              angvel_reward + contact_bonus + smoothness_penalty - action_cost)

    return reward