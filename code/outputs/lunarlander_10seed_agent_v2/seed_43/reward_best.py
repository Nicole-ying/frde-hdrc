def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large changes in absolute values
    # This encourages the agent to reduce extreme states
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_change * 0.1
    
    # Component 2: Smoothness penalty - penalize large jumps in state
    # This encourages smooth, controlled movements
    diff_squared = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -diff_squared * 0.05
    
    # Component 3: Action penalty - penalize excessive action usage
    # Convert action to a small cost
    action_cost = -0.01 * float(action)
    
    # Component 4: Ground contact bonus - encourage stable contact
    # Using info if available, otherwise from next_obs indices 6 and 7
    ground_contact_bonus = 0.0
    if len(n) >= 8:
        ground_contact_bonus = (n[6] + n[7]) * 0.5
    
    # Stage-based weighting
    # Early training: focus on movement and smoothness
    # Late training: focus on ground contact and stability
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Combine components with stage weights
    reward = (
        early_weight * (movement_reward + smoothness_penalty + action_cost) +
        late_weight * ground_contact_bonus
    )
    
    return reward