def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # Feature 1: Change in absolute values (directional toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Feature 2: Action penalty (small action cost to encourage efficiency)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.25

    # Feature 3: Contact bonus (from observation)
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = n[6] + n[7]  # Sum of leg contact indicators

    # Feature 4: Survival bonus - reward staying alive
    survival_bonus = 0.05

    # Feature 5: Velocity penalty - directional toward zero
    velocity_penalty = 0.0
    if len(o) >= 4:
        vel_abs_change = (abs(o[2]) - abs(n[2])) + (abs(o[3]) - abs(n[3]))
        velocity_penalty = vel_abs_change * 0.3

    # Feature 6: Angular velocity penalty - directional toward zero
    angular_vel_penalty = 0.0
    if len(o) >= 6:
        ang_abs_change = abs(o[5]) - abs(n[5])
        angular_vel_penalty = ang_abs_change * 0.4

    # Feature 7: Smoothness penalty - penalize large jumps (undirected, but useful for stability)
    squared_change = 0.0
    if len(o) >= 2:
        squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))

    # Stage-based weights that evolve with training progress
    if training_progress < 0.3:
        # Early exploration phase - focus on stability and survival
        w_abs = 1.0
        w_action = -0.2
        w_contact = 0.2
        w_survival = 1.0
        w_velocity = 0.5
        w_angular = 0.5
        w_smooth = -0.05
    elif training_progress < 0.6:
        # Middle refinement phase - balance all components
        w_abs = 0.8
        w_action = -0.3
        w_contact = 0.6
        w_survival = 0.6
        w_velocity = 0.4
        w_angular = 0.4
        w_smooth = -0.1
    elif training_progress < 0.85:
        # Late precision phase - emphasize contact and smoothness
        w_abs = 0.5
        w_action = -0.4
        w_contact = 1.2
        w_survival = 0.3
        w_velocity = 0.3
        w_angular = 0.3
        w_smooth = -0.2
    else:
        # Final convergence phase - maximize contact, minimize penalties
        w_abs = 0.3
        w_action = -0.5
        w_contact = 2.0
        w_survival = 0.2
        w_velocity = 0.2
        w_angular = 0.2
        w_smooth = -0.3

    # Combine components
    reward = (w_abs * abs_change + 
              w_action * action_cost + 
              w_contact * contact_bonus +
              w_survival * survival_bonus +
              w_velocity * velocity_penalty +
              w_angular * angular_vel_penalty +
              w_smooth * squared_change)

    return reward