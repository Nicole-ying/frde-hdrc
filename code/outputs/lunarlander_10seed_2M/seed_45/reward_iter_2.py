def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs

    # Component 1: Movement toward zero (stability) - directional
    # Reward reduction in absolute values across all state dimensions
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.1

    # Component 2: Smoothness penalty - penalize large state changes
    # This encourages smooth transitions
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.05

    # Component 3: Action cost - small penalty for using engines
    # Convert discrete action to a small cost
    action_cost = -0.01 * (1.0 if action != 0 else 0.0)

    # Component 4: Contact bonus - reward gaining ground contact
    # Last two dimensions are contact indicators (0 or 1)
    contact_bonus = 0.0
    if len(o) >= 8:
        prev_contact = o[-2] + o[-1]
        curr_contact = n[-2] + n[-1]
        # Reward gaining contact (landing legs touching ground)
        contact_bonus = max(0.0, curr_contact - prev_contact) * 0.5

    # Stage-based weighting
    # Early training: focus on movement and exploration
    # Mid training: balance smoothness and contact
    # Late training: emphasize precision and contact

    if training_progress < 0.3:
        # Early stage - encourage exploration and movement
        w_movement = 1.0
        w_smoothness = 0.3
        w_action = 0.5
        w_contact = 0.2
    elif training_progress < 0.7:
        # Mid stage - balance components
        w_movement = 0.7
        w_smoothness = 0.7
        w_action = 0.3
        w_contact = 0.6
    else:
        # Late stage - emphasize precision and contact
        w_movement = 0.3
        w_smoothness = 1.0
        w_action = 0.1
        w_contact = 1.0

    # Combine components
    reward = (w_movement * movement_reward + 
              w_smoothness * smoothness_penalty + 
              w_action * action_cost + 
              w_contact * contact_bonus)

    return reward