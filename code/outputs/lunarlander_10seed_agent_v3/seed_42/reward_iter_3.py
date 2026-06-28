def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement toward center - sum of absolute position changes
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))
    
    # Feature 2: Velocity reduction - sum of squared velocity differences
    vel_change = sum((n[i] - o[i]) ** 2 for i in range(2, min(6, len(o))))
    
    # Feature 3: Angular stability - change in angular components
    ang_change = sum(abs(o[i]) - abs(n[i]) for i in range(4, min(6, len(o))))
    
    # Feature 4: Contact bonus - ground contact signals
    contact_bonus = 0.0
    if len(o) >= 8:
        contact_bonus = sum(n[6:8]) - sum(o[6:8])
    
    # Feature 5: Action cost (penalize large actions)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = -0.5
    elif action in [1, 3]:  # Side engines
        action_cost = -0.2
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance stability and control
    # Late stage: refine precision and contacts
    
    # Sigmoid-like transition for smooth weight evolution
    early_weight = max(0.0, 1.0 - training_progress * 2.0)
    mid_weight = max(0.0, 1.0 - abs(training_progress - 0.5) * 2.0)
    late_weight = max(0.0, (training_progress - 0.5) * 2.0)
    
    # Component weights
    w_pos = 1.0 * early_weight + 0.5 * mid_weight + 0.2 * late_weight
    w_vel = 0.3 * early_weight + 0.8 * mid_weight + 0.5 * late_weight
    w_ang = 0.2 * early_weight + 0.6 * mid_weight + 0.8 * late_weight
    w_contact = 0.0 * early_weight + 0.3 * mid_weight + 1.0 * late_weight
    w_action = 0.1 * early_weight + 0.2 * mid_weight + 0.3 * late_weight
    
    # Combine components
    reward = (
        w_pos * pos_change * 0.1 +
        w_vel * (-vel_change * 0.01) +
        w_ang * ang_change * 0.05 +
        w_contact * contact_bonus * 0.5 +
        w_action * action_cost
    )
    
    # Add small exploration bonus early on
    exploration_bonus = 0.0
    if training_progress < 0.3:
        exploration_bonus = 0.01 * sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    reward += exploration_bonus
    
    return reward