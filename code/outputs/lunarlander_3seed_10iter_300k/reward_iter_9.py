def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays from observations
    o = obs
    n = next_obs
    
    # Component 1: Movement magnitude change (encourages reducing absolute state values)
    # Sum of absolute values difference - positive when moving toward zero
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.1
    
    # Component 2: Smoothness penalty - penalize large changes in state
    # Use squared differences to penalize erratic movements
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.05
    
    # Component 3: Action penalty - discourage excessive action usage
    # Discrete actions: penalize non-zero actions (action 0 is typically no-op)
    action_cost = 0.0
    if action != 0:
        action_cost = -0.02
    
    # Component 4: Contact bonus - reward ground contact (leg sensors at indices 6 and 7)
    # This encourages landing behavior
    left_contact = n[6]
    right_contact = n[7]
    contact_bonus = (left_contact + right_contact) * 0.5
    
    # Component 5: Velocity reduction bonus - reward reducing speed
    # Velocity components are at indices 2 and 3
    old_speed = abs(o[2]) + abs(o[3])
    new_speed = abs(n[2]) + abs(n[3])
    speed_reduction = (old_speed - new_speed) * 0.2
    
    # Combine components with stage weights based on training_progress
    # Early training: focus on movement and smoothness
    # Late training: focus on landing (contact) and speed reduction
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    total_reward = (
        movement_reward * early_weight +
        smoothness_penalty * early_weight +
        action_cost +
        contact_bonus * late_weight * 2.0 +
        speed_reduction * late_weight * 1.5
    )
    
    return total_reward