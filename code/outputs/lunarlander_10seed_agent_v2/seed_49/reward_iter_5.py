def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability (reducing absolute values)
    # Positive when absolute values decrease
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.1
    
    # Component 2: Smoothness - penalize large changes in state
    # Negative when state changes are large
    state_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -state_change * 0.05
    
    # Component 3: Action cost - penalize taking actions
    # Small penalty for any action to encourage efficiency
    action_cost = -0.01
    
    # Component 4: Ground contact bonus (from info or obs)
    # Check if legs are in contact (indices 6 and 7 in obs)
    leg_contact = 0.0
    if len(o) >= 8:
        leg_contact = (n[6] + n[7]) * 0.5  # Both legs contact gives max bonus
    
    # Component 5: Velocity reduction bonus
    # Penalize high velocities (indices 2 and 3 for x,y velocity)
    vel_penalty = 0.0
    if len(o) >= 4:
        prev_vel = abs(o[2]) + abs(o[3])
        curr_vel = abs(n[2]) + abs(n[3])
        vel_penalty = (prev_vel - curr_vel) * 0.2
    
    # Stage-based weighting
    # Early: focus on movement and exploration
    # Middle: balance smoothness and contact
    # Late: emphasize stability and ground contact
    stage1_weight = max(0.0, 1.0 - training_progress * 2.0)  # Decays in first half
    stage2_weight = 1.0 - abs(training_progress - 0.5) * 2.0  # Peaks at middle
    stage3_weight = max(0.0, training_progress * 2.0 - 1.0)  # Grows in second half
    
    # Combine components with stage weights
    reward = (
        movement_reward * (0.3 + 0.2 * stage1_weight) +
        smoothness_penalty * (0.2 + 0.3 * stage2_weight) +
        action_cost * (0.1 + 0.1 * stage1_weight) +
        leg_contact * (0.2 + 0.4 * stage3_weight) +
        vel_penalty * (0.2 + 0.2 * stage2_weight)
    )
    
    return reward