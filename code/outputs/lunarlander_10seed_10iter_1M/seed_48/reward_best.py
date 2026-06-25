def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability - penalize large changes in position/velocity
    # Use sum of squared differences as a generic smoothness signal
    delta = n - o
    movement_penalty = 0.0
    for i in range(len(delta)):
        movement_penalty += delta[i] * delta[i]
    
    # Component 2: Encourage reaching states with smaller absolute values (closer to zero)
    # This is a generic regularization signal
    state_magnitude = 0.0
    for i in range(len(n)):
        state_magnitude += abs(n[i])
    
    # Component 3: Action penalty to discourage excessive action usage
    action_penalty = 0.0
    if action == 2:  # Main engine
        action_penalty = 0.5
    elif action in [1, 3]:  # Side engines
        action_penalty = 0.2
    
    # Component 4: Contact bonus - encourage ground contact (leg sensors are indices 6 and 7)
    contact_bonus = 0.0
    if len(n) >= 8:
        contact_bonus = n[6] + n[7]  # Sum of leg contact indicators
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on movement and state magnitude
    # Late training: focus on contact and reduce action penalty
    stage1_weight = 1.0 - training_progress  # Decreases from 1 to 0
    stage2_weight = training_progress  # Increases from 0 to 1
    
    # Combine components with stage weights
    reward = 0.0
    
    # Early stage: penalize movement and state magnitude
    reward -= stage1_weight * 0.1 * movement_penalty
    reward -= stage1_weight * 0.05 * state_magnitude
    
    # Late stage: encourage contact and reduce action penalty
    reward += stage2_weight * 2.0 * contact_bonus
    reward -= stage2_weight * 0.1 * action_penalty
    
    # Add a small constant to encourage exploration
    reward += 0.01
    
    return reward