def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Unpack observations
    o = obs
    n = next_obs
    
    # Current state
    x, y = o[0], o[1]
    vx, vy = o[2], o[3]
    angle = o[4]
    angular_vel = o[5]
    leg1_contact = o[6]
    leg2_contact = o[7]
    
    # Next state
    nx, ny = n[0], n[1]
    nvx, nvy = n[2], n[3]
    nangle = n[4]
    nangular_vel = n[5]
    nleg1_contact = n[6]
    nleg2_contact = n[7]
    
    # Dynamic weights based on training progress
    if training_progress < 0.3:
        # Early: prioritize exploration, gentle shaping
        w_distance = 0.3
        w_velocity = 0.2
        w_angle = 0.1
        w_angular = 0.1
        w_fuel = 0.1
        w_landing = 0.1
        w_crash = 0.1
    elif training_progress < 0.7:
        # Middle: balance all components
        w_distance = 0.25
        w_velocity = 0.2
        w_angle = 0.15
        w_angular = 0.1
        w_fuel = 0.1
        w_landing = 0.1
        w_crash = 0.1
    else:
        # Late: precision and stability
        w_distance = 0.2
        w_velocity = 0.15
        w_angle = 0.2
        w_angular = 0.15
        w_fuel = 0.05
        w_landing = 0.15
        w_crash = 0.1
    
    # 1. Distance to landing pad (0,0)
    dist_to_pad = (nx**2 + ny**2) ** 0.5
    dist_reward = -dist_to_pad  # Negative distance, want to minimize
    
    # 2. Velocity penalty (want soft landing)
    speed = (nvx**2 + nvy**2) ** 0.5
    # Penalize high speed, especially vertical speed
    vertical_speed_penalty = max(0, nvy)  # Positive vy means moving up (bad for landing)
    horizontal_speed_penalty = abs(nvx)
    velocity_reward = -(speed + 2.0 * vertical_speed_penalty + 0.5 * horizontal_speed_penalty)
    
    # 3. Angle penalty (want upright)
    angle_penalty = -abs(nangle) * 2.0
    # Extra penalty for large angles (risk of crash)
    if abs(nangle) > 0.5:
        angle_penalty -= 5.0 * (abs(nangle) - 0.5)
    
    # 4. Angular velocity penalty (want stable)
    angular_vel_penalty = -abs(nangular_vel) * 3.0
    
    # 5. Fuel efficiency (penalize engine use)
    # Action 2 is main engine, actions 1 and 3 are side engines
    fuel_penalty = 0.0
    if action == 2:  # Main engine
        fuel_penalty = -0.5
    elif action == 1 or action == 3:  # Side engines
        fuel_penalty = -0.2
    
    # 6. Landing reward
    landing_reward = 0.0
    both_legs_contact = nleg1_contact > 0.5 and nleg2_contact > 0.5
    on_pad = dist_to_pad < 0.1
    
    if both_legs_contact and on_pad:
        # Perfect landing
        landing_reward = 10.0
        # Bonus for low speed
        if speed < 0.5:
            landing_reward += 5.0
        # Bonus for upright
        if abs(nangle) < 0.1:
            landing_reward += 3.0
    elif both_legs_contact:
        # Touched ground but not on pad
        landing_reward = 1.0
        # Penalty if far from pad
        if dist_to_pad > 0.3:
            landing_reward -= 2.0
    
    # 7. Crash penalty
    crash_penalty = 0.0
    # Detect crash: high speed with ground contact, or extreme angle
    if both_legs_contact and speed > 2.0:
        crash_penalty = -5.0
    if abs(nangle) > 1.0:  # Nearly upside down
        crash_penalty = -8.0
    
    # Progress reward: getting closer to pad
    prev_dist = (x**2 + y**2) ** 0.5
    progress_reward = (prev_dist - dist_to_pad) * 2.0
    
    # Combine all rewards
    total_reward = (
        w_distance * dist_reward +
        w_velocity * velocity_reward +
        w_angle * angle_penalty +
        w_angular * angular_vel_penalty +
        w_fuel * fuel_penalty +
        w_landing * landing_reward +
        w_crash * crash_penalty +
        progress_reward
    )
    
    return total_reward