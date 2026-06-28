def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement toward stability - penalize large changes in absolute values
    # This encourages smooth transitions and reduces oscillation
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_quality = abs_diff / max(len(o), 1)
    
    # Component 2: State change magnitude - penalize large sudden changes
    # Small changes indicate controlled behavior
    state_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_reward = -state_change / max(len(o), 1)
    
    # Component 3: Action cost - penalize frequent/aggressive actions
    # Actions are discrete (0-3), so we penalize non-zero actions
    action_cost = -0.1 * (1.0 if action != 0 else 0.0)
    
    # Component 4: Terminal signal from info (if available)
    # Check for termination conditions that might indicate success/failure
    terminal_bonus = 0.0
    if info:
        # Generic check for any boolean signals in info
        for key, value in info.items():
            if isinstance(value, bool) and value:
                terminal_bonus += 0.5
    
    # Stage-based weighting
    # Early training: focus on smoothness and movement quality
    # Late training: focus on terminal outcomes and action efficiency
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)  # Early stage
    stage2_weight = min(1.0, 2.0 * training_progress)  # Late stage
    
    # Combine components with stage weights
    reward = (
        stage1_weight * (0.5 * movement_quality + 0.3 * smoothness_reward) +
        stage2_weight * (0.4 * terminal_bonus + 0.3 * action_cost) +
        0.2 * (movement_quality + smoothness_reward + action_cost + terminal_bonus) / 4.0
    )
    
    return reward