def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as arrays
    o = obs
    n = next_obs

    # --- SKELETON QUALITY DIAGNOSIS ---
    # Step 1: Current components:
    #   - abs_diff: sum(abs(o) - abs(n)) — directional movement toward zero
    #   - sq_diff: sum((n-o)^2) — undirected change magnitude (redundant with abs_diff)
    #   - action_cost: small penalty per action type
    # Step 2: Missing signals:
    #   - No survival bonus (episode length 119 → agent dies fast)
    #   - No velocity/speed penalty (crashing at high speed)
    #   - No contact/landing reward (leg ground contact)
    #   - No angle/angular velocity stability signal
    # Step 3: Issues:
    #   - sq_diff is UNDIRECTED — rewards ANY change, not just desirable ones
    #   - abs_diff and sq_diff are REDUNDANT (both measure state-change magnitude)
    #   - No positive reward for staying alive or making progress
    #   - Action penalty is tiny and constant — not helping
    # Step 4: REBUILD — score is very negative, episode length very short, skeleton clearly insufficient

    # --- NEW SKELETON ---
    # Use the unmasked step context to extract meaningful signals:
    # obs[0]: x position (normalized)
    # obs[1]: y position (normalized, 0=helipad)
    # obs[2]: x velocity
    # obs[3]: y velocity
    # obs[4]: angle
    # obs[5]: angular velocity
    # obs[6]: left leg contact
    # obs[7]: right leg contact

    # Survival bonus — reward for staying alive
    survival_bonus = 0.1

    # Progress toward center (x=0) — directional
    x_progress = abs(o[0]) - abs(n[0])

    # Progress toward landing pad (y=0) — directional
    y_progress = abs(o[1]) - abs(n[1])

    # Speed penalty — discourage high velocity (crashing)
    speed_penalty = (o[2]**2 + o[3]**2)  # squared speed

    # Angle stability — penalize large angles and angular velocity
    angle_penalty = abs(o[4]) + 0.1 * abs(o[5])

    # Contact reward — reward for having legs on ground
    contact_reward = 0.0
    if len(o) >= 8:
        contact_reward = o[6] + o[7]  # sum of leg contacts (0, 1, or 2)

    # Action cost — discourage excessive engine use
    if action == 0:
        action_cost = 0.0
    elif action == 2:
        action_cost = 0.2  # main engine — expensive
    else:
        action_cost = 0.1  # side engines — moderate

    # Stage-based weights
    early_weight = 1.0 - training_progress
    late_weight = training_progress

    # Component weights that evolve with training
    w_survival = 0.3 * early_weight + 0.5 * late_weight
    w_x = 0.4 * early_weight + 0.2 * late_weight
    w_y = 0.5 * early_weight + 0.3 * late_weight
    w_speed = 0.1 * early_weight + 0.4 * late_weight
    w_angle = 0.1 * early_weight + 0.4 * late_weight
    w_contact = 0.0 * early_weight + 0.3 * late_weight  # only matters late
    w_action = 0.2 * early_weight + 0.2 * late_weight

    # Compute reward components
    reward_survival = survival_bonus * w_survival
    reward_x = x_progress * w_x
    reward_y = y_progress * w_y
    reward_speed = -speed_penalty * w_speed
    reward_angle = -angle_penalty * w_angle
    reward_contact = contact_reward * w_contact
    reward_action = -action_cost * w_action

    # Combine components
    reward = (reward_survival + reward_x + reward_y +
              reward_speed + reward_angle +
              reward_contact + reward_action)

    return float(reward)