def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs

    # Transition progress: reward moving toward zero velocity and center position
    # Use directional signal: abs(o) - abs(n) for position (indices 0,1) and velocity (indices 2,3)
    progress_pos = (abs(o[0]) - abs(n[0])) + (abs(o[1]) - abs(n[1]))
    progress_vel = (abs(o[2]) - abs(n[2])) + (abs(o[3]) - abs(n[3]))
    transition_progress = progress_pos + progress_vel

    # Smoothness penalty: penalize large changes in any state component
    smoothness = sum(abs(n[i] - o[i]) for i in range(8))

    # Landing reward: reward both legs contacting ground
    ground_contact = n[6] + n[7]

    # Angle penalty: penalize deviation from upright (angle index 4)
    angle_penalty = abs(n[4])

    # Angular velocity penalty: penalize spinning (index 5)
    angular_vel_penalty = abs(n[5])

    # Action cost: penalize engine use
    action_cost = 0.0
    if action == 2:
        action_cost = 1.0
    elif action in [1, 3]:
        action_cost = 0.5

    # Survival bonus: encourage staying alive longer
    survival_bonus = 0.1

    reward = (
        2.0 * transition_progress
        - 0.1 * smoothness
        + 5.0 * ground_contact
        - 2.0 * angle_penalty
        - 1.0 * angular_vel_penalty
        - 0.5 * action_cost
        + survival_bonus
    )

    return reward