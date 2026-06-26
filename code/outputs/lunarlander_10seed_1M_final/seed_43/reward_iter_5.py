def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # --- Signal 1: Movement toward origin (directional) ---
    # Encourages reducing absolute values of state components
    movement_toward_origin = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # --- Signal 2: Velocity penalty (from obs indices 2,3) ---
    # Penalizes high speed to prevent crashes
    vel_penalty = 0.0
    if len(o) >= 4:
        vel_penalty = (o[2] ** 2 + o[3] ** 2)

    # --- Signal 3: Angle penalty (from obs index 4) ---
    # Penalizes large angles to prevent tumbling
    angle_penalty = 0.0
    if len(o) >= 5:
        angle_penalty = abs(o[4])

    # --- Signal 4: Angular velocity penalty (from obs index 5) ---
    # Penalizes spinning
    angular_vel_penalty = 0.0
    if len(o) >= 6:
        angular_vel_penalty = abs(o[5])

    # --- Signal 5: Action cost ---
    action_cost = 0.0
    if action == 2:
        action_cost = 1.0
    elif action in [1, 3]:
        action_cost = 0.5

    # --- Signal 6: Contact bonus (from obs indices 6,7) ---
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = o[6] + o[7]

    # --- Signal 7: Height change toward ground (from obs index 1) ---
    # obs[1] is normalized y-position (positive = above ground, negative = below)
    # Encourages moving downward (toward ground) in early/mid training
    height_change = 0.0
    if len(o) >= 2:
        height_change = n[1] - o[1]  # positive = moving up, negative = moving down
        # We want negative values (moving down), so we'll use -height_change as reward

    # Stage-based weights that evolve with training_progress
    # Early stage: focus on movement toward origin and downward movement
    # Middle stage: balance movement with stability
    # Late stage: prioritize smooth landing and contact

    if training_progress < 0.3:
        # Early training: encourage movement and descent, very low penalties
        w_movement = 1.5
        w_vel = -0.05
        w_angle = -0.05
        w_angvel = -0.02
        w_action = -0.05
        w_contact = 0.0
        w_height = -0.3  # reward moving downward (negative height_change)
    elif training_progress < 0.7:
        # Middle training: balance movement with stability
        w_movement = 1.0
        w_vel = -0.2
        w_angle = -0.2
        w_angvel = -0.1
        w_action = -0.15
        w_contact = 0.3
        w_height = -0.2  # still reward descent but less emphasis
    else:
        # Late training: prioritize smooth landing and contact
        w_movement = 0.5
        w_vel = -0.4
        w_angle = -0.4
        w_angvel = -0.2
        w_action = -0.25
        w_contact = 0.8
        w_height = 0.0  # no height reward in late stage (contact is more important)

    # Combine components
    reward = (w_movement * movement_toward_origin +
              w_vel * vel_penalty +
              w_angle * angle_penalty +
              w_angvel * angular_vel_penalty +
              w_action * action_cost +
              w_contact * contact_bonus +
              w_height * height_change)

    return reward