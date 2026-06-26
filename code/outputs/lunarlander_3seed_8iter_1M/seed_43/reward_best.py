def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Feature 1: Movement toward center - encourage reducing absolute values
    # Sum of absolute value changes (negative means moving toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Feature 2: Smoothness - penalize large changes in state
    diff_sq = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -0.01 * diff_sq

    # Feature 3: Action penalty - discourage excessive action usage
    # For discrete actions, penalize non-zero actions slightly
    action_cost = -0.02 if action != 0 else 0.0

    # Feature 4: Contact bonus - if legs make contact (indices 6 and 7 in observation)
    # This is a generic signal that contact events might be beneficial
    contact_bonus = 0.0
    if len(o) >= 8:
        left_leg_contact = n[6] > 0.5
        right_leg_contact = n[7] > 0.5
        if left_leg_contact and right_leg_contact:
            contact_bonus = 0.5
        elif left_leg_contact or right_leg_contact:
            contact_bonus = 0.2

    # Feature 5: Stability - penalize angular velocity and angle
    # Indices 4 (angle) and 5 (angular velocity) in observation
    stability_penalty = 0.0
    if len(o) >= 6:
        angle_penalty = -0.1 * abs(n[4])
        ang_vel_penalty = -0.05 * abs(n[5])
        stability_penalty = angle_penalty + ang_vel_penalty

    # Feature 6: Velocity penalty - discourage high speeds (indices 2 and 3)
    # High velocities can cause instability and crashes
    velocity_penalty = 0.0
    if len(o) >= 4:
        velocity_penalty = -0.02 * (abs(n[2]) + abs(n[3]))

    # Stage weights based on training progress
    # Early: focus on exploration and movement
    # Middle: balance movement and stability
    # Late: focus on precision and contact
    if training_progress < 0.3:
        # Early stage - encourage exploration and movement
        w_abs = 1.0
        w_smooth = 0.1
        w_action = 0.5
        w_contact = 0.0
        w_stability = 0.1
        w_velocity = 0.0
    elif training_progress < 0.7:
        # Middle stage - balance all components
        w_abs = 0.8
        w_smooth = 0.3
        w_action = 0.3
        w_contact = 0.3
        w_stability = 0.3
        w_velocity = 0.2
    else:
        # Late stage - focus on precision and landing
        w_abs = 0.5
        w_smooth = 0.5
        w_action = 0.2
        w_contact = 0.8
        w_stability = 0.6
        w_velocity = 0.4

    # Combine components
    reward = (w_abs * abs_change + 
              w_smooth * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus + 
              w_stability * stability_penalty +
              w_velocity * velocity_penalty)

    return reward