def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    o = obs
    n = next_obs
    
    # Current state
    x, y, vx, vy, angle, angular_vel, leg1_contact, leg2_contact = o
    
    # Next state
    nx, ny, nvx, nvy, nangle, nangular_vel, nleg1_contact, nleg2_contact = n
    
    # Check if crashed or out of bounds (terminal conditions)
    # x > 1 means out of viewport, y < -1.5 means crashed (rough estimate)
    crashed = ny < -1.5 or abs(nx) > 1.0
    
    # Landing detection: both legs on ground, near pad, low velocity, upright
    both_legs_contact = nleg1_contact > 0.5 and nleg2_contact > 0.5
    near_pad = abs(nx) < 0.1 and abs(ny) < 0.1
    low_velocity = abs(nvx) < 0.1 and abs(nvy) < 0.1
    upright = abs(nangle) < 0.1
    landed = both_legs_contact and near_pad and low_velocity and upright
    
    # Dynamic weights based on training progress
    if training_progress < 0.3:
        # Early: encourage exploration, gentle penalties
        w_survival = 1.0
        w_progress = 0.5
        w_fuel = 0.1
        w_stability = 0.2
        w_precision = 0.1
    elif training_progress < 0.7:
        # Middle: balance all objectives
        w_survival = 2.0
        w_progress = 1.0
        w_fuel = 0.3
        w_stability = 0.5
        w_precision = 0.5
    else:
        # Late: prioritize precision and stability
        w_survival = 3.0
        w_progress = 1.0
        w_fuel = 0.2
        w_stability = 1.0
        w_precision = 2.0
    
    # 1. Survival reward (avoid crashing)
    survival_reward = 0.0
    if not crashed:
        survival_reward = 1.0
    else:
        survival_reward = -10.0
    
    # 2. Progress toward landing pad
    # Distance to pad (0,0) in next state
    dist_to_pad = (nx**2 + ny**2) ** 0.5
    # Previous distance
    prev_dist = (x**2 + y**2) ** 0.5
    # Improvement in distance
    distance_improvement = prev_dist - dist_to_pad
    progress_reward = distance_improvement * 2.0  # Scale up
    
    # 3. Fuel efficiency (penalize engine use)
    fuel_penalty = 0.0
    if action == 2:  # Main engine
        fuel_penalty = -0.3
    elif action in [1, 3]:  # Side engines
        fuel_penalty = -0.1
    
    # 4. Stability reward
    # Penalize large angles and angular velocity
    angle_penalty = -abs(nangle) * 2.0
    angular_vel_penalty = -abs(nangular_vel) * 0.5
    stability_reward = angle_penalty + angular_vel_penalty
    
    # 5. Precision reward for landing
    precision_reward = 0.0
    if both_legs_contact:
        # Reward being close to pad with low velocity when legs touch
        precision_reward = (1.0 - min(dist_to_pad, 1.0)) * 2.0
        precision_reward += (1.0 - min(abs(nvx) + abs(nvy), 1.0)) * 2.0
        precision_reward += (1.0 - min(abs(nangle), 1.0)) * 2.0
    
    # 6. Landing bonus
    landing_bonus = 0.0
    if landed:
        landing_bonus = 50.0
    
    # Combine all rewards
    total_reward = (
        w_survival * survival_reward +
        w_progress * progress_reward +
        w_fuel * fuel_penalty +
        w_stability * stability_reward +
        w_precision * precision_reward +
        landing_bonus
    )
    
    return total_reward