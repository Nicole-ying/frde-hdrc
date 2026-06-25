def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays for generic processing
    o = obs
    n = next_obs

    # Feature 1: Directional movement toward zero (encourages stabilization)
    # Kept from best iteration (score 29.959) - showed consistent improvement
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Feature 2: Smoothness penalty - penalize large changes between steps
    smoothness = sum(abs(n[i] - o[i]) for i in range(len(o)))

    # Feature 3: Action cost - penalize engine usage
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3

    # Feature 4: Velocity change signal - penalize rapid velocity changes
    vel_change = abs(n[2] - o[2]) + abs(n[3] - o[3])

    # Feature 5: Stability signal - penalize angular velocity
    angular_vel_penalty = abs(n[5]) if len(o) > 5 else 0.0

    # Feature 6: Contact bonus - reward ground contact
    contact_bonus = 0.0
    if len(o) > 7:
        contact_bonus = n[6] + n[7]

    # Feature 7: Height signal - reward moving toward ground (assuming obs[1] is height)
    height_change = abs(o[1]) - abs(n[1]) if len(o) > 1 else 0.0

    # Feature 8: Horizontal position signal - reward centering (assuming obs[0] is x position)
    horiz_change = abs(o[0]) - abs(n[0]) if len(o) > 0 else 0.0

    # Feature 9: Survival bonus - reward staying alive each step
    survival_bonus = 0.1

    # Stage weights based on training_progress
    # Using weights from best iteration (iter 3, score 29.959) with minor tuning
    # Removed state_mag_change and terminal_bonus which degraded performance in iter 4

    if training_progress < 0.3:
        # Early stage: encourage exploration and basic movement
        w_abs = 0.8
        w_smooth = -0.1
        w_action = -0.05
        w_vel = -0.05
        w_angular = -0.2
        w_contact = 0.0
        w_height = 0.8
        w_horiz = 0.5
        w_survival = 0.2
    elif training_progress < 0.7:
        # Middle stage: balance exploration with stability
        w_abs = 0.5
        w_smooth = -0.2
        w_action = -0.1
        w_vel = -0.1
        w_angular = -0.4
        w_contact = 0.5
        w_height = 1.0
        w_horiz = 0.7
        w_survival = 0.3
    else:
        # Late stage: focus on precision and landing
        w_abs = 0.3
        w_smooth = -0.3
        w_action = -0.2
        w_vel = -0.2
        w_angular = -0.8
        w_contact = 2.0
        w_height = 1.5
        w_horiz = 1.0
        w_survival = 0.5

    # Combine components
    reward = (w_abs * abs_change +
              w_smooth * smoothness +
              w_action * action_cost +
              w_vel * vel_change +
              w_angular * angular_vel_penalty +
              w_contact * contact_bonus +
              w_height * height_change +
              w_horiz * horiz_change +
              w_survival * survival_bonus)

    return reward