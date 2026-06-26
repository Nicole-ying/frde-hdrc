def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Generic transition features from observation arrays
    o = obs
    n = next_obs
    
    # Feature 1: Sum of absolute differences (encourages movement in early stages)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change magnitude (captures overall transition size)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (discourages excessive engine use)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = -0.5
    elif action in [1, 3]:  # Side engines
        action_cost = -0.2
    
    # Feature 4: Ground contact bonus (from info or observation)
    # Using last two observation dimensions which indicate leg contact
    leg_contact = 0.0
    if len(o) >= 2:
        leg_contact = sum(n[-2:]) - sum(o[-2:])  # Change in contact state
    
    # Stage-based weighting
    stage1 = min(1.0, training_progress * 3.0)  # Early exploration
    stage2 = min(1.0, max(0.0, (training_progress - 0.3) * 3.0))  # Mid refinement
    stage3 = min(1.0, max(0.0, (training_progress - 0.6) * 3.0))  # Late exploitation
    
    # Component weights evolve with training
    w_abs_diff = 0.5 * (1.0 - stage1) + 0.1 * stage1  # Decrease over time
    w_sq_change = 0.3 * stage1 + 0.5 * stage2 + 0.2 * stage3  # Peak in mid training
    w_action_cost = 0.2 + 0.3 * stage2 + 0.5 * stage3  # Increase over time
    w_contact = 0.0 * stage1 + 0.2 * stage2 + 0.5 * stage3  # Late stage focus
    
    # Combine components
    reward = (
        w_abs_diff * abs_diff +
        w_sq_change * sq_change +
        w_action_cost * action_cost +
        w_contact * leg_contact
    )
    
    return reward