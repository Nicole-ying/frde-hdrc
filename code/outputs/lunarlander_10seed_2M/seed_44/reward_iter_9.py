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

    # Component 4: Penalize large state changes (smoothness) - use L1 norm
    vel_change = sum(abs(n[i] - o[i]) for i in range(len(o)))

    # Component 5: Small action penalty to encourage efficiency
    action_cost = 0.01 * (1 if action != 0 else 0)

    # Component 6: Reward for maintaining ground contact (staying landed)
    # This is different from new_contact - it rewards keeping legs on ground
    contact_maintained = 0.0
    if len(n) > 6 and len(o) > 6:
        contact_maintained = (n[6] + n[7]) * 0.5  # Average contact of both legs

    # Component 7: Penalize vertical velocity (index 3) to encourage soft landing
    vert_vel = abs(n[3]) if len(n) > 3 else 0.0

    # Component 8: Reward for being close to center horizontally (index 0)
    horiz_pos = abs(n[0]) if len(n) > 0 else 0.0

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
        w6 = 0.0  # No contact maintenance reward early
        w7 = 0.2  # Light vertical velocity penalty
        w8 = 0.5  # Moderate horizontal position reward
    elif training_progress < 0.7:
        w1 = 1.0  # Moderate abs_change
        w2 = 0.6  # Increasing angular velocity penalty
        w3 = 1.5  # Stronger contact reward
        w4 = 0.3  # Moderate smoothness
        w5 = 0.2  # Moderate action penalty
        w6 = 0.5  # Some contact maintenance reward
        w7 = 0.5  # Moderate vertical velocity penalty
        w8 = 0.3  # Less horizontal position reward
    else:
        w1 = 0.5  # Less abs_change late
        w2 = 1.0  # Strong angular velocity penalty
        w3 = 3.0  # Strong contact reward
        w4 = 0.5  # Stronger smoothness
        w5 = 0.3  # Stronger action penalty
        w6 = 1.0  # Strong contact maintenance reward
        w7 = 1.0  # Strong vertical velocity penalty
        w8 = 0.1  # Minimal horizontal position reward

    # Combine components
    reward = (w1 * abs_change + 
              w2 * (-ang_vel * 0.5) + 
              w3 * (new_contact * 2.0) + 
              w4 * (-vel_change * 0.1) + 
              w5 * (-action_cost) +
              w6 * (contact_maintained * 1.0) +
              w7 * (-vert_vel * 0.3) +
              w8 * (-horiz_pos * 0.5))

    return reward