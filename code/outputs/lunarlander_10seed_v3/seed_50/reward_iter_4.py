def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Component 1: Height-based reward - reward being higher (index 1 is y-position, negative = low)
    # This encourages the agent to stay in the air longer
    height_reward = n[1] - o[1]  # Positive when going up
    
    # Component 2: Vertical velocity penalty - discourage fast downward movement (index 3 is y-velocity)
    # Negative velocity = moving down, we want to penalize this
    vert_vel_penalty = max(0, -n[3])  # Only penalize downward velocity
    
    # Component 3: Horizontal velocity penalty - discourage sideways drift (index 2 is x-velocity)
    horiz_vel_penalty = abs(n[2])
    
    # Component 4: Angle penalty - discourage tilting (index 4)
    angle_penalty = abs(n[4])
    
    # Component 5: Angular velocity penalty - discourage spinning (index 5)
    ang_vel_penalty = abs(n[5])
    
    # Component 6: Leg contact bonus - reward ground contact (indices 6,7)
    leg_contact = n[6] + n[7]
    
    # Component 7: Action penalty - penalize engine use
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 1.0
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # Stage-based weights - completely restructured
    # Early: focus on staying alive (height, gentle penalties)
    # Late: focus on landing (leg contact, stability)
    
    # Height reward - strong early to keep agent alive, decreases later
    w_height = 3.0 * (1.0 - 0.7 * training_progress)
    
    # Vertical velocity penalty - moderate throughout, increases slightly
    w_vert_vel = 2.0 * (1.0 + 0.5 * training_progress)
    
    # Horizontal velocity penalty - moderate
    w_horiz_vel = 1.0
    
    # Angle penalty - moderate, increases late for stable landing
    w_angle = 1.0 * (1.0 + 0.5 * training_progress)
    
    # Angular velocity penalty - moderate
    w_ang_vel = 0.5 * (1.0 + 0.5 * training_progress)
    
    # Leg contact - only matters late
    w_leg = 5.0 * training_progress
    
    # Action cost - moderate early to allow exploration, increases late
    w_action = 0.5 * (1.0 + training_progress)
    
    # Survival bonus - constant to keep agent alive
    survival_bonus = 0.5
    
    reward = (w_height * height_reward
              - w_vert_vel * vert_vel_penalty
              - w_horiz_vel * horiz_vel_penalty
              - w_angle * angle_penalty
              - w_ang_vel * ang_vel_penalty
              + w_leg * leg_contact
              - w_action * action_cost
              + survival_bonus)
    
    return reward