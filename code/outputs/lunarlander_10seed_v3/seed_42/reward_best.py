def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Generic transition signals from observation arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large changes in absolute values
    # This encourages smooth transitions and reaching stable states
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_change * 0.1
    
    # Component 2: Velocity/change penalty - penalize large squared differences
    # This discourages erratic movements and encourages precision
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    change_penalty = -squared_diff * 0.05
    
    # Component 3: Action cost - penalize taking actions to encourage efficiency
    # Normalize action to [0,1] range for discrete actions
    action_cost = -0.01 * (action / 3.0)
    
    # Component 4: Terminal state bonus - check if legs are in contact (indices 6,7)
    # This encourages reaching ground contact states
    leg_contact_bonus = 0.0
    if len(o) >= 8:
        leg_contact_bonus = (n[6] + n[7]) * 0.5
    
    # Stage-based weighting using training_progress
    # Early training: focus on exploration and movement
    # Mid training: balance movement and stability
    # Late training: focus on precision and terminal states
    stage_weight_1 = 1.0 - 0.5 * training_progress  # Movement weight decreases
    stage_weight_2 = 0.3 + 0.7 * training_progress   # Change penalty increases
    stage_weight_3 = 1.0 - 0.8 * training_progress   # Action cost decreases
    stage_weight_4 = 0.0 + 1.0 * training_progress   # Leg contact bonus increases
    
    # Combine components with stage weights
    reward = (
        stage_weight_1 * movement_reward +
        stage_weight_2 * change_penalty +
        stage_weight_3 * action_cost +
        stage_weight_4 * leg_contact_bonus
    )
    
    return reward