def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation components
    o = obs
    n = next_obs

    # Feature 1: Change in absolute values (encourages movement toward zero)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # Feature 2: Action penalty (small cost for taking actions)
    action_cost = 0.01 * abs(action)

    # Feature 3: Velocity change (from observation indices 2 and 3)
    vel_change = (abs(o[2]) - abs(n[2])) + (abs(o[3]) - abs(n[3]))

    # Feature 4: Ground contact bonus (from observation indices 6 and 7)
    ground_contact = n[6] + n[7]

    # Feature 5: Stability signal (from observation indices 4 and 5 - angle and angular velocity)
    angle_stability = abs(o[4]) - abs(n[4])
    angular_vel_stability = abs(o[5]) - abs(n[5])
    stability = angle_stability + angular_vel_stability

    # Feature 6: Survival bonus (encourages longer episodes)
    survival_bonus = 0.1

    # Stage-based weights
    # Early training: focus on exploration and velocity reduction
    # Mid training: balance between position and velocity
    # Late training: focus on precision and landing

    if training_progress < 0.3:
        # Early stage: encourage movement and exploration
        w_abs = 1.0
        w_action = -0.1
        w_vel = 0.5
        w_ground = 0.0
        w_stability = 0.3
        w_survival = 0.2
    elif training_progress < 0.7:
        # Mid stage: balance
        w_abs = 0.5
        w_action = -0.05
        w_vel = 1.0
        w_ground = 0.5
        w_stability = 0.5
        w_survival = 0.3
    else:
        # Late stage: precision and landing
        w_abs = 0.2
        w_action = -0.02
        w_vel = 2.0
        w_ground = 2.0
        w_stability = 1.0
        w_survival = 0.5

    # Combine components
    reward = (w_abs * abs_change + 
              w_action * action_cost + 
              w_vel * vel_change + 
              w_ground * ground_contact +
              w_stability * stability +
              w_survival * survival_bonus)

    return reward