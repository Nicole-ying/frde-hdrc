def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs

    # Transition progress: reward moving toward zero position and velocity (directional)
    pos_vel_progress = 0.0
    for i in range(4):
        pos_vel_progress += abs(o[i]) - abs(n[i])

    # Angular stability: reward reducing angle and angular velocity
    angle_progress = abs(o[4]) - abs(n[4])
    angvel_progress = abs(o[5]) - abs(n[5])

    # Ground contact bonus: reward new contacts
    ground_contact_next = n[6] + n[7]
    ground_contact_prev = o[6] + o[7]
    new_contacts = max(0.0, ground_contact_next - ground_contact_prev)

    # Action penalty (discrete 0-3)
    action_cost = 0.0
    if action == 2:  # main engine
        action_cost = 0.5
    elif action in [1, 3]:  # side engines
        action_cost = 0.2

    # Survival bonus: reward staying alive
    survival_bonus = 0.01

    # Reward components with balanced scales - simplified to avoid conflicting signals
    reward = (
        +2.0 * pos_vel_progress
        +1.0 * angle_progress
        +0.5 * angvel_progress
        +3.0 * new_contacts
        -0.3 * action_cost
        + survival_bonus
    )

    return reward