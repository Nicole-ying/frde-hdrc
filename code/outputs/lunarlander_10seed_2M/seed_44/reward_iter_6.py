def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Encourage reduction in absolute state values (approach toward zero/center)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    component_1 = abs_diff
    
    # Component 2: Penalize large state changes (smoothness)
    delta_sq = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    component_2 = -delta_sq * 0.1
    
    # Component 3: Small action penalty to encourage efficiency
    # action is discrete (0-3), penalize any action slightly
    component_3 = -0.01 * (1 if action != 0 else 0)
    
    # Component 4: Reward for ground contact (leg sensors at indices 6 and 7)
    leg_contact_prev = 1.0 if len(o) > 6 and o[6] > 0.5 else 0.0
    leg_contact_next = 1.0 if len(n) > 6 and n[6] > 0.5 else 0.0
    leg_contact_prev2 = 1.0 if len(o) > 7 and o[7] > 0.5 else 0.0
    leg_contact_next2 = 1.0 if len(n) > 7 and n[7] > 0.5 else 0.0
    
    # Reward for gaining ground contact
    new_contact = max(0, (leg_contact_next + leg_contact_next2) - (leg_contact_prev + leg_contact_prev2))
    component_4 = new_contact * 2.0
    
    # Component 5: Penalize angular velocity (index 5) to encourage stability
    ang_vel = abs(n[5]) if len(n) > 5 else 0.0
    component_5 = -ang_vel * 0.5
    
    # Stage weights based on training_progress
    # Early stage: focus on exploration and getting to center
    # Middle stage: balance smoothness and contact
    # Late stage: prioritize landing (contact) and stability
    
    if training_progress < 0.3:
        w1 = 1.0
        w2 = 0.3
        w3 = 0.1
        w4 = 0.5
        w5 = 0.2
    elif training_progress < 0.7:
        w1 = 0.8
        w2 = 0.6
        w3 = 0.2
        w4 = 1.0
        w5 = 0.5
    else:
        w1 = 0.3
        w2 = 0.8
        w3 = 0.3
        w4 = 2.0
        w5 = 1.0
    
    # Combine components
    reward = (w1 * component_1 + 
              w2 * component_2 + 
              w3 * component_3 + 
              w4 * component_4 + 
              w5 * component_5)
    
    return reward