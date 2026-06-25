def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition signals
    # 1. Movement magnitude (squared differences) - encourages exploration
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 2. Directional improvement - reward moving toward desirable states (abs decreasing)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 3. Smoothness penalty - penalize large changes in state (jittery behavior)
    smoothness = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 4. Action cost - penalize engine usage
    action_cost = 0.0
    if action == 2:  # main engine
        action_cost = 0.2
    elif action in [1, 3]:  # side engines
        action_cost = 0.1
    
    # 5. Stability signal - penalize rapid changes (angular velocity proxy via state changes)
    # Use the difference between consecutive state changes as a smoothness measure
    stability = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Stage-based weights
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Component rewards with proper scaling
    # Movement reward: encourage exploration early, discourage late
    movement_reward = movement * 0.3
    
    # Directional improvement: always positive when moving toward zero
    abs_change_reward = abs_change * 0.5
    
    # Smoothness penalty: penalize jerky movements (same as movement but negative)
    smoothness_penalty = -smoothness * 0.1
    
    # Stability bonus: reward small changes (inverse of movement)
    stability_bonus = -movement * 0.05  # small penalty for any movement
    
    # Action penalty
    action_penalty = -action_cost
    
    # Survival bonus: reward staying alive (longer episodes)
    survival_bonus = 0.02  # small constant per step
    
    # Combine with stage weights
    # Early: explore with movement, but also reward directional improvement
    # Mid: balance exploration and exploitation
    # Late: focus on stability and convergence
    reward = (
        early_weight * (movement_reward + abs_change_reward + survival_bonus) +
        mid_weight * (abs_change_reward * 0.7 + movement_reward * 0.3 + survival_bonus) +
        late_weight * (abs_change_reward * 0.5 - movement * 0.2 + survival_bonus) +
        action_penalty + smoothness_penalty + stability_bonus
    )
    
    return reward