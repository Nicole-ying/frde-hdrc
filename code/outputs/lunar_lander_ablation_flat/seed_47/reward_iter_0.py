def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Movement penalty - encourage staying still (small velocity changes)
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Stability reward - encourage smooth transitions
    stability = sum((n[i] - o[i]) ** 2 for i in range(8))
    
    # Ground contact bonus - encourage landing
    ground_contact = n[6] + n[7]
    
    # Angle penalty - encourage upright landing
    angle_penalty = abs(n[4])
    
    # Action penalty - encourage efficient fuel use
    action_cost = 0.0
    if action == 2:
        action_cost = 0.1
    elif action in [1, 3]:
        action_cost = 0.05
    
    reward = (
        0.5 * movement
        - 0.1 * stability
        + 2.0 * ground_contact
        - 0.5 * angle_penalty
        - 0.2 * action_cost
    )
    
    return reward