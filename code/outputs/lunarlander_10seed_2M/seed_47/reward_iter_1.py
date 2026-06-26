def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components as generic arrays
    o = obs
    n = next_obs

    # --- Signal 1: Position toward center (directional) ---
    # Use first 2 dims (x, y position) - reward moving toward origin
    pos_x_change = abs(o[0]) - abs(n[0]) if len(o) > 0 else 0.0
    pos_y_change = abs(o[1]) - abs(n[1]) if len(o) > 1 else 0.0
    pos_signal = pos_x_change + pos_y_change

    # --- Signal 2: Velocity damping (directional) ---
    # Use dims 2,3 (x,y velocity) - reward reducing speed
    vel_x_change = abs(o[2]) - abs(n[2]) if len(o) > 2 else 0.0
    vel_y_change = abs(o[3]) - abs(n[3]) if len(o) > 3 else 0.0
    vel_signal = vel_x_change + vel_y_change

    # --- Signal 3: Angle stabilization (directional) ---
    # Dim 4 is angle - reward moving angle toward 0
    angle_signal = abs(o[4]) - abs(n[4]) if len(o) > 4 else 0.0

    # --- Signal 4: Angular velocity damping (directional) ---
    # Dim 5 is angular velocity - reward reducing spin
    angvel_signal = abs(o[5]) - abs(n[5]) if len(o) > 5 else 0.0

    # --- Signal 5: Ground contact bonus ---
    # Dims 6,7 are leg contact indicators - reward new contacts
    contact_signal = 0.0
    if len(o) >= 8:
        contact_signal = (n[6] - o[6]) + (n[7] - o[7])

    # --- Signal 6: Action penalty (discourage unnecessary engine use) ---
    # For discrete actions, penalize main engine (action==2) and side engines (action in [1,3])
    action_penalty = 0.0
    if action == 2:  # main engine
        action_penalty = -1.0
    elif action in [1, 3]:  # side engines
        action_penalty = -0.5

    # --- Stage weights ---
    # Early: explore, learn to move and stabilize
    # Middle: refine positioning and landing
    # Late: precision landing with contacts
    early_w = max(0.0, 1.0 - 2.0 * training_progress)
    mid_w = 1.0 - abs(2.0 * training_progress - 1.0)
    late_w = max(0.0, 2.0 * training_progress - 1.0)

    # --- Combine with stage-appropriate scaling ---
    # Early: focus on position, velocity, angle, angular velocity
    early_reward = (
        1.0 * pos_signal
        + 0.8 * vel_signal
        + 0.6 * angle_signal
        + 0.4 * angvel_signal
        + 0.2 * action_penalty
    )

    # Middle: continue stabilization, start caring about contacts
    mid_reward = (
        0.8 * pos_signal
        + 0.6 * vel_signal
        + 0.5 * angle_signal
        + 0.3 * angvel_signal
        + 0.5 * contact_signal
        + 0.3 * action_penalty
    )

    # Late: emphasize contacts and precise positioning
    late_reward = (
        0.5 * pos_signal
        + 0.3 * vel_signal
        + 0.3 * angle_signal
        + 0.2 * angvel_signal
        + 1.5 * contact_signal
        + 0.1 * action_penalty
    )

    # Combine stages
    reward = early_w * early_reward + mid_w * mid_reward + late_w * late_reward

    # Small survival bonus to encourage longer episodes
    reward += 0.05

    return float(reward)