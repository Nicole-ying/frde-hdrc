def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Component 1: Stability - penalize large angles and angular velocity
    angle = abs(o[4])
    ang_vel = abs(o[5])
    stability_penalty = angle + ang_vel
    
    # Component 2: Velocity penalty - discourage high speed crashes
    vel_x = abs(o[2])
    vel_y = abs(o[3])
    velocity_penalty = vel_x + vel_y
    
    # Component 3: Transition progress - reward movement toward center/zero
    progress = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Component 4: Ground contact bonus
    ground_contact = o[-1] + o[-2]
    
    # Component 5: Survival bonus - reward staying alive
    survival_bonus = 1.0
    
    # Component 6: Action cost
    action_cost = 0.0
    if action == 2:
        action_cost = 0.5
    elif action in [1, 3]:
        action_cost = 0.2
    
    # Stage weights
    if training_progress < 0.3:
        w_stability = -0.5
        w_velocity = -0.3
        w_progress = 1.0
        w_contact = 0.3
        w_survival = 0.2
        w_action = -0.1
    elif training_progress < 0.7:
        w_stability = -1.0
        w_velocity = -0.5
        w_progress = 0.8
        w_contact = 0.8
        w_survival = 0.5
        w_action = -0.3
    else:
        w_stability = -2.0
        w_velocity = -1.0
        w_progress = 0.5
        w_contact = 2.0
        w_survival = 1.0
        w_action = -0.5
    
    reward = (w_stability * stability_penalty +
              w_velocity * velocity_penalty +
              w_progress * progress +
              w_contact * ground_contact +
              w_survival * survival_bonus +
              w_action * action_cost)
    
    return reward