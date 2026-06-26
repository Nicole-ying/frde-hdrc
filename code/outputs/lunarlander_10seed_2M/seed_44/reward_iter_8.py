def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Component 1: Movement toward zero (directional - encourages stabilization)
    # This was the only consistently positive component across all iterations
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Component 2: Penalize angular velocity (index 5) to encourage stability
    # Added in iteration 6 and correlated with score improvement
    ang_vel = abs(n[5]) if len(n) > 5 else 0.0

    # Component 3: Reward for ground contact (leg sensors at indices 6 and 7)
    # Added in iteration 6 and correlated with score improvement
    leg_contact_prev = 1.0 if len(o) > 6 and o[6] > 0.5 else 0.0
    leg_contact_next = 1.0 if len(n) > 6 and n[6] > 0.5 else 0.0
    leg_contact_prev2 = 1.0 if len(o) > 7 and o[7] > 0.5 else 0.0
    leg_contact_next2 = 1.0 if len(n) > 7 and n[7] > 0.5 else 0.0
    new_contact = max(0, (leg_contact_next + leg_contact_next2) - (leg_contact_prev + leg_contact_prev2))

    # Component 4: Penalize large state changes (smoothness) - use L1 norm instead of squared
    # L1 is more robust and less punishing than squared changes
    vel_change = sum(abs(n[i] - o[i]) for i in range(len(o)))

    # Component 5: Small action penalty to encourage efficiency
    action_cost = 0.01 * (1 if action != 0 else 0)

    # Stage weights based on training_progress
    # Early stage: focus on exploration and getting to center
    # Middle stage: balance smoothness and contact
    # Late stage: prioritize landing (contact) and stability

    if training_progress < 0.3:
        w1 = 2.0  # Strong abs_change signal early
        w2 = 0.3  # Light angular velocity penalty
        w3 = 0.5  # Some contact reward
        w4 = 0.1  # Light smoothness penalty
        w5 = 0.1  # Light action penalty
    elif training_progress < 0.7:
        w1 = 1.0  # Moderate abs_change
        w2 = 0.6  # Increasing angular velocity penalty
        w3 = 1.5  # Stronger contact reward
        w4 = 0.3  # Moderate smoothness
        w5 = 0.2  # Moderate action penalty
    else:
        w1 = 0.5  # Less abs_change late
        w2 = 1.0  # Strong angular velocity penalty
        w3 = 3.0  # Strong contact reward
        w4 = 0.5  # Stronger smoothness
        w5 = 0.3  # Stronger action penalty

    # Combine components
    reward = (w1 * abs_change + 
              w2 * (-ang_vel * 0.5) + 
              w3 * (new_contact * 2.0) + 
              w4 * (-vel_change * 0.1) + 
              w5 * (-action_cost))

    return reward