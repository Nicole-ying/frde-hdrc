def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement toward center - sum of absolute position changes
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))
    
    # Feature 2: Velocity reduction - sum of squared velocity changes
    vel_change = sum((n[i] - o[i]) ** 2 for i in range(2, min(6, len(o))))
    
    # Feature 3: Stability - penalize large angular changes
    ang_change = abs(n[4] - o[4]) if len(o) > 4 else 0.0
    
    # Feature 4: Contact bonus - ground contact signals
    contact_bonus = 0.0
    if len(o) > 6:
        contact_bonus = sum(n[6:8]) - sum(o[6:8])
    
    # Feature 5: Action cost - penalize engine usage
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # Stage-based weights that evolve with training_progress
    # Stage 1 (early): Focus on position stabilization and contact
    # Stage 2 (mid): Balance velocity control and stability
    # Stage 3 (late): Fine-tune with action efficiency
    
    w_pos = 1.0 + 2.0 * training_progress  # Increases importance of position
    w_vel = 1.0 - 0.5 * training_progress  # Decreases velocity penalty
    w_ang = 0.5 + 0.5 * training_progress   # Increases angular stability
    w_contact = 0.5 + 1.5 * training_progress  # Increases contact importance
    w_action = 0.1 + 0.4 * training_progress  # Increases action efficiency
    
    # Compute reward components
    reward_pos = w_pos * pos_change
    reward_vel = -w_vel * vel_change * 0.1  # Small negative for velocity changes
    reward_ang = -w_ang * ang_change * 0.5  # Penalize angular changes
    reward_contact = w_contact * contact_bonus * 2.0  # Bonus for ground contact
    reward_action = -w_action * action_cost  # Penalize action usage
    
    # Combine components
    reward = reward_pos + reward_vel + reward_ang + reward_contact + reward_action
    
    # Add small exploration bonus early in training
    exploration_bonus = 0.01 * (1.0 - training_progress)
    reward += exploration_bonus
    
    return reward