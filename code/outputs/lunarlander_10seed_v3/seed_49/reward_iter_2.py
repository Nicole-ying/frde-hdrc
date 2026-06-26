def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Feature 1: Movement toward zero (directional - rewards reducing absolute values)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Velocity penalty (penalizes high speeds - prevents crashing)
    # obs[2] = x velocity, obs[3] = y velocity
    vel_penalty = abs(o[2]) + abs(o[3])
    
    # Feature 3: Angular stability (penalizes spinning)
    # obs[4] = angle, obs[5] = angular velocity
    angle_penalty = abs(o[4]) + abs(o[5])
    
    # Feature 4: Landing encouragement (reward ground contact)
    # obs[6] and obs[7] are leg contact indicators
    contact_bonus = o[6] + o[7]
    
    # Feature 5: Action cost (penalizes large actions)
    action_cost = float(action) ** 2
    
    # Stage weights based on training progress
    # Early: focus on stability and reducing absolute values
    # Late: focus on landing and fine control
    w1 = 1.0 - 0.5 * training_progress  # abs_change weight
    w2 = 0.5 + 0.5 * training_progress   # velocity penalty weight (increase over time)
    w3 = 0.5 + 0.5 * training_progress   # angle penalty weight (increase over time)
    w4 = 0.0 + 1.0 * training_progress   # contact bonus weight (only late)
    w5 = 0.01 * (1.0 - training_progress)  # action cost weight
    
    reward = (w1 * abs_change 
              - w2 * vel_penalty 
              - w3 * angle_penalty 
              + w4 * contact_bonus 
              - w5 * action_cost)
    
    return reward