def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # ========== SIGNAL INVENTORY ==========
    # 1. Transition progress: movement toward desirable states
    # 2. Smoothness: penalize jerky changes
    # 3. Action cost: penalize engine usage
    # 4. Survival bonus: reward staying alive
    # 5. Velocity penalty: discourage high speeds (crashes)
    # 6. Contact bonus: reward ground contact (landing)
    
    # ========== COMPUTE RAW SIGNALS ==========
    
    # Signal 1: Transition progress - reduction in absolute values (directional)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Signal 2: Smoothness - penalize large changes (undirected, but we want smooth)
    sq_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Signal 3: Action cost
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 1.0
    elif action in [1, 3]:  # Side engines
        action_cost = 0.5
    
    # Signal 4: Survival bonus - constant positive for each step alive
    survival_bonus = 0.1
    
    # Signal 5: Velocity penalty - detect high speeds from obs/n
    # obs[2] and obs[3] are likely velocity components (from step source)
    # Penalize large absolute velocities
    vel_x = abs(o[2]) if len(o) > 2 else 0.0
    vel_y = abs(o[3]) if len(o) > 3 else 0.0
    velocity_penalty = (vel_x + vel_y) * 0.2
    
    # Signal 6: Contact bonus - reward ground contact
    # obs[6] and obs[7] are leg contact indicators (from step source)
    contact_left = o[6] if len(o) > 6 else 0.0
    contact_right = o[7] if len(o) > 7 else 0.0
    contact_bonus = (contact_left + contact_right) * 0.5
    
    # ========== STAGE WEIGHTS ==========
    # Early: explore, focus on survival and reducing abs values
    # Middle: balance smoothness, velocity control, and contact
    # Late: refine with action efficiency and landing
    
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # ========== COMPONENT REWARDS ==========
    # Scale each component to contribute ~0.1-1.0 per step
    
    # Transition progress: reward moving toward zero (stability)
    stability_reward = abs_diff * 0.3  # Positive when improving
    
    # Smoothness: penalize large changes
    smoothness_reward = -sq_diff * 0.05  # Mild penalty
    
    # Action penalty: discourage unnecessary engine use
    action_penalty = -action_cost * 0.2
    
    # Survival: constant positive reward
    survival_reward = survival_bonus * 1.0
    
    # Velocity penalty: discourage high speeds (prevents crashes)
    vel_penalty = -velocity_penalty * 0.5
    
    # Contact bonus: reward being on ground
    contact_reward = contact_bonus * 1.0
    
    # ========== COMBINE WITH STAGE WEIGHTS ==========
    # Early stage: survival + stability (exploration)
    early_reward = (
        stability_reward * 0.4 +
        survival_reward * 0.4 +
        vel_penalty * 0.1 +
        contact_reward * 0.1
    )
    
    # Middle stage: balance everything
    mid_reward = (
        stability_reward * 0.2 +
        smoothness_reward * 0.2 +
        survival_reward * 0.2 +
        vel_penalty * 0.2 +
        contact_reward * 0.2
    )
    
    # Late stage: focus on landing (contact) and efficiency
    late_reward = (
        contact_reward * 0.3 +
        action_penalty * 0.2 +
        smoothness_reward * 0.2 +
        stability_reward * 0.2 +
        survival_reward * 0.1
    )
    
    # Weighted combination
    weight_sum = early_weight + mid_weight + late_weight
    if weight_sum > 0:
        reward = (early_weight * early_reward + mid_weight * mid_reward + late_weight * late_reward) / weight_sum
    else:
        reward = 0.0
    
    # Add small constant to prevent zero reward
    reward = reward + 0.01
    
    return reward