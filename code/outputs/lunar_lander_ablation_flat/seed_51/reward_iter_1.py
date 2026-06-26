def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs

    # Transition progress: reward moving toward zero position and velocity (directional)
    # Use signed change toward zero for position (dim 0-1) and velocity (dim 2-3)
    pos_vel_progress = 0.0
    for i in range(4):
        pos_vel_progress += abs(o[i]) - abs(n[i])

    # Angular stability: reward reducing angle and angular velocity (dim 4-5)
    angle_progress = abs(o[4]) - abs(n[4])
    angvel_progress = abs(o[5]) - abs(n[5])

    # Ground contact bonus: reward new contacts (dim 6-7)
    ground_contact_next = n[6] + n[7]
    ground_contact_prev = o[6] + o[7]
    new_contacts = max(0.0, ground_contact_next - ground_contact_prev)

    # Action penalty (discrete 0-3)
    action_cost = 0.0
    if action == 2:  # main engine
        action_cost = 0.5
    elif action in [1, 3]:  # side engines
        action_cost = 0.2

    # Survival bonus: reward staying alive (longer episodes)
    survival_bonus = 0.01

    # Reward components with balanced scales
    reward = (
        +1.0 * pos_vel_progress
        +0.5 * angle_progress
        +0.3 * angvel_progress
        +2.0 * new_contacts
        -0.3 * action_cost
        +0.1 * (1.0 - abs(n[4]))  # encourage upright orientation
        -0.2 * abs(n[1])  # penalize distance from landing pad height
        + survival_bonus
    )

    return reward