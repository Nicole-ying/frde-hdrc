def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # 1. Movement magnitude: encourage any state change (exploration)
    movement = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 2. Action penalty: small cost for taking actions to prevent thrashing
    action_cost = 0.01 * (1.0 if action != 0 else 0.0)
    
    # 3. Contact bonus: reward leg ground contact (indices 6,7)
    left_leg = n[6] if len(n) > 6 else 0.0
    right_leg = n[7] if len(n) > 7 else 0.0
    contact_bonus = 0.5 * (left_leg + right_leg)
    
    # 4. Stability: penalize large angle and angular velocity (indices 4,5)
    angle_penalty = -0.1 * abs(n[4]) if len(n) > 4 else 0.0
    angvel_penalty = -0.05 * abs(n[5]) if len(n) > 5 else 0.0
    
    # 5. Velocity penalty: penalize high speed (indices 2,3 are velocity)
    vel_x = n[2] if len(n) > 2 else 0.0
    vel_y = n[3] if len(n) > 3 else 0.0
    speed = (vel_x**2 + vel_y**2) ** 0.5
    velocity_penalty = -0.1 * speed
    
    # 6. Height bonus: reward being lower (closer to ground) - index 1 is y-position
    height = n[1] if len(n) > 1 else 0.0
    height_bonus = -0.2 * height  # positive when height is negative (lower)
    
    # 7. Survival bonus: reward staying alive longer (counteract early termination)
    survival_bonus = 0.5  # constant positive reward per step
    
    # Stage weights
    progress = max(0.0, min(1.0, training_progress))
    early_weight = 1.0 - 0.8 * progress
    late_weight = 0.2 + 0.8 * progress
    
    # Combine: early focus on movement/exploration, late focus on stability/contact/height
    # Add survival bonus to keep agent alive longer
    reward = (
        early_weight * (0.5 * movement - action_cost)
        + late_weight * (angle_penalty + angvel_penalty + velocity_penalty + contact_bonus + height_bonus)
        + survival_bonus
    )
    
    return reward