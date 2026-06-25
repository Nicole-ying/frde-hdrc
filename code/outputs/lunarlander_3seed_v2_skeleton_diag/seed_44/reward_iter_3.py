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
    # This is a constant positive reward per step that increases with training progress
    survival_bonus = 0.1 * (1.0 + training_progress)

    # Feature 6: Progress signal - reward reduction in observation magnitude
    # This is a denser version of abs_change that works better
    progress_signal = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Stage weights based on training progress
    # Early exploration: encourage movement and exploration
    # Late exploitation: encourage stability and precision
    w1 = 1.0 - 0.5 * training_progress  # weight for abs_change (decreases over time)
    w2 = 0.1 + 0.5 * training_progress  # weight for stability (increases over time)
    w3 = 0.05  # constant action penalty
    w4 = 0.2 * training_progress  # weight for velocity penalty (increases over time)
    w5 = 1.0  # survival bonus weight
    w6 = 0.5 * (1.0 - training_progress)  # weight for progress signal (decreases over time)

    # Combine components - simplified and rebalanced
    reward = (w1 * abs_change 
              - w2 * sq_change 
              - w3 * action_cost 
              - w4 * vel_change 
              + w5 * survival_bonus
              + w6 * progress_signal)

    return reward