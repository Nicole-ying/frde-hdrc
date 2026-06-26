def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Component 1: Movement toward stability - reward when absolute values decrease
    # This is the most consistently effective signal across iterations
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    convergence_bonus = abs_diff * 0.1

    # Component 2: Velocity penalty - penalize high speeds (prevents crashing)
    # obs[2] = x velocity, obs[3] = y velocity
    vel_penalty = abs(o[2]) + abs(o[3])

    # Component 3: Angular stability - penalize spinning and tilting
    # obs[4] = angle, obs[5] = angular velocity
    angle_penalty = abs(o[4]) + abs(o[5])

    # Component 4: Landing encouragement - reward ground contact
    # obs[6] and obs[7] are leg contact indicators
    contact_bonus = o[6] + o[7]

    # Component 5: Action penalty - discourage excessive engine use
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.3
    elif action in [1, 3]:  # Side engines
        action_cost = 0.15

    # Stage-based weighting
    # Early stage: focus on stability (velocity + angle penalties) and convergence
    # Late stage: focus on landing (contact bonus) and fine control
    w_convergence = 1.0 - 0.3 * training_progress  # Decrease slightly over time
    w_stability = 0.8 - 0.4 * training_progress  # Decrease over time
    w_landing = 0.8 * training_progress  # Increase over time
    w_action = 0.2 * (1.0 - 0.5 * training_progress)  # Decrease over time

    # Combine components
    reward = (
        w_convergence * convergence_bonus
        - w_stability * vel_penalty
        - w_stability * angle_penalty
        + w_landing * contact_bonus
        - w_action * action_cost
    )

    return float(reward)