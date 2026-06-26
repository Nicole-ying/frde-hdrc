def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Component 1: Movement toward stability - penalize large changes in absolute values
    # This encourages the agent to reduce extreme positions/velocities
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.1

    # Component 2: Smoothness penalty - penalize large squared changes
    # This encourages smooth transitions and discourages erratic behavior
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.05

    # Component 3: Action cost - penalize taking actions to encourage efficiency
    action_cost = -0.01 * float(action)

    # Component 4: Contact bonus - reward ground contact (from info or observation)
    # Last two dimensions of obs indicate ground contact (leg sensors)
    contact_bonus = 0.0
    if len(o) >= 8:
        leg1_contact = o[6]
        leg2_contact = o[7]
        next_leg1_contact = n[6]
        next_leg2_contact = n[7]
        # Reward maintaining or gaining ground contact
        contact_bonus = (next_leg1_contact + next_leg2_contact) * 0.5

    # Stage weights based on training progress
    # Early stage: focus on movement and smoothness
    # Middle stage: balance all components
    # Late stage: emphasize contact and efficiency

    if training_progress < 0.3:
        # Early exploration: prioritize movement and smoothness
        w_movement = 1.0
        w_smoothness = 0.8
        w_action = 0.3
        w_contact = 0.2
    elif training_progress < 0.6:
        # Middle stage: balanced weights
        w_movement = 0.8
        w_smoothness = 0.6
        w_action = 0.5
        w_contact = 0.6
    else:
        # Late stage: focus on contact and efficiency
        w_movement = 0.4
        w_smoothness = 0.3
        w_action = 0.7
        w_contact = 1.0

    # Combine components with stage weights
    reward = (w_movement * movement_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus)

    return reward