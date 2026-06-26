def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs

    # Position components (indices 0,1)
    # Velocity components (indices 2,3)
    # Angle and angular velocity (indices 4,5)
    # Ground contact flags (indices 6,7)

    # --- Transition signals (directional) ---

    # Reward for moving toward landing pad (x=0)
    pos_x_diff = abs(o[0]) - abs(n[0])

    # Reward for moving downward (negative y velocity means descending)
    # We want the lander to go down, so negative y velocity is good
    # n[3] is current y velocity, o[3] is previous y velocity
    # If descending (negative), we want to maintain or increase descent
    # Positive vel_y_change means slowing descent (bad for landing)
    # Negative vel_y_change means speeding up descent (good initially)
    vel_y_change = n[3] - o[3]  # negative means increasing downward speed

    # Reward for reducing velocity magnitude (safer landing)
    vel_mag_old = (o[2]**2 + o[3]**2)**0.5
    vel_mag_new = (n[2]**2 + n[3]**2)**0.5
    vel_reduction = vel_mag_old - vel_mag_new

    # Reward for upright orientation (angle close to 0)
    angle_penalty = -abs(n[4])

    # Reward for reducing angular velocity
    angvel_reduction = abs(o[5]) - abs(n[5])

    # Reward for ground contact (landing legs touching)
    ground_contact_reward = n[6] + n[7]

    # --- New: Survival bonus to encourage longer episodes ---
    # Small constant reward for staying alive
    survival_bonus = 0.05

    # --- New: Penalty for being too high (encourage descent) ---
    # y position (index 1) is normalized, positive means above helipad
    height_penalty = -abs(n[1]) * 0.1  # Penalize being far from helipad height

    # --- New: Penalty for high horizontal speed ---
    # x velocity (index 2) should be small for stable landing
    horizontal_speed_penalty = -abs(n[2]) * 0.2

    # --- Action penalty (encourage efficient use of engines) ---
    # action 0: do nothing, action 1: left, action 2: main, action 3: right
    action_cost = 0.0
    if action == 2:  # main engine
        action_cost = -0.15
    elif action == 1 or action == 3:  # side engines
        action_cost = -0.08

    # Combine rewards with fixed coefficients
    reward = (
        1.0 * pos_x_diff +            # Move toward center (stronger)
        0.5 * vel_y_change +          # Control descent rate (stronger)
        0.8 * vel_reduction +         # Reduce speed (stronger)
        -0.5 * angle_penalty +        # Stay upright (stronger)
        0.3 * angvel_reduction +      # Reduce spin (stronger)
        3.0 * ground_contact_reward + # Touch ground with legs (stronger)
        survival_bonus +              # Stay alive bonus
        height_penalty +              # Encourage descending
        horizontal_speed_penalty +    # Penalize horizontal drift
        action_cost                   # Engine usage cost
    )

    return reward