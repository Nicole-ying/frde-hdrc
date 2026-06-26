def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs

    # --- Signal 1: Progress toward center (directional) ---
    # Encourages moving toward zero in all state dimensions
    progress = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # --- Signal 2: Smoothness penalty (penalizes large jumps) ---
    # Use L1 norm to penalize abrupt changes
    smoothness = -sum(abs(n[i] - o[i]) for i in range(len(o)))

    # --- Signal 3: Action cost (penalize engine use) ---
    # Discrete action: 0=noop, 1=left, 2=main, 3=right
    if action == 0:
        action_cost = 0.0
    elif action == 2:
        action_cost = 0.3  # main engine is expensive
    else:
        action_cost = 0.15  # side engines

    # --- Signal 4: Ground contact bonus ---
    # Last two dims are leg contact indicators (0 or 1)
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = float(n[6] + n[7])  # 0, 1, or 2 legs on ground

    # --- Signal 5: Velocity penalty (discourage high speed near ground) ---
    # Dims 2 and 3 are x and y velocity
    vel_penalty = 0.0
    if len(o) >= 4:
        # Penalize high absolute velocity in any direction
        vel_penalty = -(abs(n[2]) + abs(n[3])) * 0.5

    # --- Stage-based weights ---
    if training_progress < 0.3:
        # Early: explore, move toward center, allow some action
        w_progress = 1.0
        w_smooth = 0.2
        w_action = -0.1
        w_contact = 0.0
        w_vel = 0.0
    elif training_progress < 0.7:
        # Middle: balance movement, smoothness, start caring about contact
        w_progress = 0.6
        w_smooth = 0.4
        w_action = -0.2
        w_contact = 0.5
        w_vel = -0.3
    else:
        # Late: precision landing, avoid crashing, maximize contact
        w_progress = 0.3
        w_smooth = 0.3
        w_action = -0.3
        w_contact = 1.5
        w_vel = -0.5

    # Combine components
    reward = (w_progress * progress +
              w_smooth * smoothness +
              w_action * action_cost +
              w_contact * contact_bonus +
              w_vel * vel_penalty)

    return reward