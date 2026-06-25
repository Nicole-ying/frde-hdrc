def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Feature 2: Squared change magnitude (penalizes large jumps, encourages smooth transitions)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # Feature 3: Action penalty (small action cost to encourage efficiency)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.25

    # Feature 4: Contact bonus (from info or observation)
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = n[6] + n[7]  # Sum of leg contact indicators

    # Feature 5: Survival bonus - reward staying alive
    survival_bonus = 0.05

    # Feature 6: Velocity penalty - penalize high velocity changes (crash prevention)
    vel_change = 0.0
    if len(o) >= 4:
        vel_change = abs(n[2] - o[2]) + abs(n[3] - o[3])
    velocity_penalty = -vel_change * 0.1

    # Feature 7: Angular velocity penalty - prevent spinning
    angular_vel_penalty = 0.0
    if len(o) >= 6:
        angular_vel_penalty = -abs(n[5]) * 0.2

    # Stage-based weights that evolve with training progress
    if training_progress < 0.3:
        # Early exploration phase
        w_abs = 1.0
        w_sq = -0.1
        w_action = -0.2
        w_contact = 0.3
        w_survival = 0.5
        w_velocity = 0.3
        w_angular = 0.3
    elif training_progress < 0.6:
        # Middle refinement phase
        w_abs = 0.8
        w_sq = -0.3
        w_action = -0.3
        w_contact = 0.8
        w_survival = 0.3
        w_velocity = 0.2
        w_angular = 0.2
    elif training_progress < 0.85:
        # Late precision phase
        w_abs = 0.5
        w_sq = -0.5
        w_action = -0.4
        w_contact = 1.5
        w_survival = 0.2
        w_velocity = 0.1
        w_angular = 0.1
    else:
        # Final convergence phase
        w_abs = 0.3
        w_sq = -0.7
        w_action = -0.5
        w_contact = 2.0
        w_survival = 0.1
        w_velocity = 0.05
        w_angular = 0.05

    # Combine components
    reward = (w_abs * abs_change + 
              w_sq * squared_change + 
              w_action * action_cost + 
              w_contact * contact_bonus +
              w_survival * survival_bonus +
              w_velocity * velocity_penalty +
              w_angular * angular_vel_penalty)

    return reward