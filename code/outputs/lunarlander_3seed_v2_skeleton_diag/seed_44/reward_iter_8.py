def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Dimension-agnostic transition features

    # Feature 1: Movement toward zero (encourages stabilization)
    # Positive when absolute values decrease
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Feature 2: Smoothness penalty (discourages large jerky changes)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # Feature 3: Action cost (penalizes engine usage)
    # For discrete actions, action is a scalar integer
    action_cost = 0.01 * float(action)

    # Feature 4: Velocity change signal (encourages slowing down)
    # Use the difference in consecutive observations as a proxy for velocity
    vel_change = sum(abs(n[i] - o[i]) for i in range(len(o)))

    # Feature 5: Survival bonus (encourages staying alive longer)
    survival_bonus = 0.1 * (1.0 + training_progress)

    # Feature 6: Angular stability signal - penalize large angular changes
    # Use the last two dimensions as proxies for angular velocity and angle
    angular_penalty = 0.0
    if len(o) >= 6:
        angular_penalty = abs(n[4]) + abs(n[5])

    # Feature 7: Ground contact bonus - reward stable landing
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = n[6] + n[7]

    # Feature 8: Height descent signal - reward moving downward (negative vertical velocity)
    # Use index 3 as vertical velocity proxy (positive = moving up, negative = moving down)
    descent_bonus = 0.0
    if len(o) >= 4:
        # Reward negative vertical velocity (moving down) and penalize positive (moving up)
        descent_bonus = -n[3]  # positive when moving down

    # Feature 9: Horizontal centering signal - reward being near center horizontally
    # Use index 0 as horizontal position proxy
    horizontal_penalty = 0.0
    if len(o) >= 1:
        horizontal_penalty = abs(n[0])

    # Stage weights based on training progress
    # Historical analysis:
    # - Iteration 7 (score=-0.140) was the best so far, but still not solving
    # - The agent survives long (979 steps) but doesn't land successfully
    # - Key insight: need to add directional signals for descent and centering
    # - abs_change and contact_bonus are working well - keep them
    # - angular_penalty is important for stability - keep it
    # - Add descent_bonus to encourage moving downward toward landing pad
    # - Add horizontal_penalty to encourage staying centered

    w1 = 1.5 - 0.5 * training_progress  # weight for abs_change (increased, decreases over time)
    w2 = 0.1 + 0.5 * training_progress  # weight for stability (increases over time)
    w3 = 0.05  # constant action penalty
    w4 = 0.1 * training_progress  # weight for velocity penalty (reduced, increases over time)
    w5 = 1.0  # survival bonus weight
    w6 = 0.3 + 0.5 * training_progress  # weight for angular penalty (reduced, increases over time)
    w7 = 0.3 + 1.0 * training_progress  # weight for contact bonus (reduced, increases over time)
    w8 = 0.5 + 1.0 * training_progress  # weight for descent bonus (increases over time)
    w9 = 0.3 + 0.5 * training_progress  # weight for horizontal penalty (increases over time)

    # Combine components - added descent and horizontal signals
    reward = (w1 * abs_change 
              - w2 * sq_change 
              - w3 * action_cost 
              - w4 * vel_change 
              + w5 * survival_bonus
              - w6 * angular_penalty
              + w7 * contact_bonus
              + w8 * descent_bonus
              - w9 * horizontal_penalty)

    return reward