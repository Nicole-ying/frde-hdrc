def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs

    # Transition progress: reward moving toward zero in x position and velocity
    progress_x = abs(o[0]) - abs(n[0])
    progress_vx = abs(o[2]) - abs(n[2])

    # Reward reducing vertical speed
    progress_vy = abs(o[3]) - abs(n[3])

    # Reward reducing angular velocity magnitude
    progress_angvel = abs(o[5]) - abs(n[5])

    # Combined transition progress signal
    transition_progress = progress_x + progress_vx + progress_vy + progress_angvel

    # Angle penalty: reward being upright (angle near zero)
    angle_penalty = abs(n[4]) * 0.5

    # Ground contact bonus: reward both legs touching ground
    ground_bonus = n[6] + n[7]

    # Action cost: small penalty for using engines
    action_cost = 0.01 * (1 if action == 2 else 0.5 if action in [1, 3] else 0.0)

    # Height penalty: reward being closer to ground (y position near zero)
    height_penalty = abs(n[1]) * 0.1

    # Survival bonus: small constant to encourage longer episodes
    survival_bonus = 0.1

    reward = (
        transition_progress * 0.5
        - angle_penalty
        + ground_bonus * 2.0
        - action_cost
        - height_penalty
        + survival_bonus
    )

    return reward