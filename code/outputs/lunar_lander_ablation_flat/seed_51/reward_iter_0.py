def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    o = obs
    n = next_obs
    
    # Position and velocity changes (dimensions 0-3)
    pos_vel_change = sum((n[i] - o[i]) ** 2 for i in range(4))
    
    # Angular velocity change (dimension 4-5)
    angle_change = abs(n[4] - o[4]) + abs(n[5] - o[5])
    
    # Ground contact signals (dimensions 6-7)
    ground_contact_next = n[6] + n[7]
    ground_contact_prev = o[6] + o[7]
    new_contacts = max(0.0, ground_contact_next - ground_contact_prev)
    
    # Action penalty (discrete 0-3)
    action_cost = 0.0
    if action == 2:  # main engine
        action_cost = 0.5
    elif action in [1, 3]:  # side engines
        action_cost = 0.2
    
    # Reward components
    reward = (
        -0.5 * pos_vel_change
        -0.1 * angle_change
        +2.0 * new_contacts
        -0.3 * action_cost
        +0.1 * (1.0 - abs(n[4]))  # encourage upright orientation
        -0.2 * abs(n[1])  # penalize distance from landing pad height
    )
    
    return reward