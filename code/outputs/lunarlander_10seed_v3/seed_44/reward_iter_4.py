def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # --- Generic transition signals ---

    # 1) Position progress: movement toward center (directional)
    pos_progress = sum(abs(o[i]) - abs(n[i]) for i in range(4))

    # 2) Velocity magnitude penalty: discourage high speed (directional)
    vel_penalty = - (abs(n[2]) + abs(n[3]))

    # 3) Angular stability: reward upright orientation (angle close to 0)
    angle_stability = -abs(n[4])

    # 4) Angular velocity penalty: discourage spinning
    ang_vel_penalty = -abs(n[5])

    # 5) Ground contact bonus: reward touching ground
    ground_contact = n[6] + n[7]

    # 6) Action cost: penalize engine use (discrete actions)
    action_cost = 0.0
    if action == 2:          # Main engine
        action_cost = 0.5
    elif action in [1, 3]:   # Side engines
        action_cost = 0.3

    # 7) Survival bonus: reward staying alive (reduced to avoid dominating)
    survival_bonus = 0.15

    # 8) Height penalty: discourage being too high (encourage descent)
    height_penalty = -abs(n[1]) * 0.2

    # 9) Velocity change smoothness: penalize jerky velocity changes
    vel_change = sum(abs(n[i] - o[i]) for i in range(2, 4))
    smoothness_penalty = -vel_change * 0.03

    # --- Stage weights (smooth transition) ---
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)

    # --- Component rewards with calibrated scales ---
    pos_reward = pos_progress * 1.2
    vel_reward = vel_penalty * 0.15
    angle_reward = angle_stability * 0.6
    ang_vel_reward = ang_vel_penalty * 0.25
    contact_reward = ground_contact * 1.5
    action_penalty = -action_cost * 0.4
    height_reward = height_penalty * 0.25
    smoothness_reward = smoothness_penalty * 0.08

    # --- Stage-specific combinations ---
    # Early: focus on position progress, survival, and height descent
    early_component = (pos_reward + survival_bonus + height_reward) * early_weight

    # Mid: balance stability, velocity control, contact, and smoothness
    mid_component = (angle_reward + ang_vel_reward + vel_reward + 
                     contact_reward + smoothness_reward) * mid_weight

    # Late: refine with contact bonus, action efficiency, and stability
    late_component = (contact_reward * 1.2 + action_penalty + 
                      angle_reward * 0.4 + vel_reward * 0.3) * late_weight

    # --- Combine ---
    total_reward = early_component + mid_component + late_component

    return float(total_reward)