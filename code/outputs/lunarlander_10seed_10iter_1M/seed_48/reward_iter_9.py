def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Component 1: Transition progress - reward movement toward zero (directional)
    # This was the only consistently positive component across all iterations
    transition_progress = 0.0
    for i in range(len(o)):
        transition_progress += abs(o[i]) - abs(n[i])
    
    # Component 2: Smoothness penalty - penalize large state changes (undirected)
    # Use squared differences but with a small coefficient to avoid dominating
    smoothness_penalty = 0.0
    for i in range(len(o)):
        smoothness_penalty += (n[i] - o[i]) * (n[i] - o[i])
    
    # Component 3: Action cost - penalize using engines
    # Based on iteration 8, action 2 (main engine) should be penalized more
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.3
    elif action in [1, 3]:  # Side engines
        action_cost = 0.1
    
    # Component 4: Contact bonus - reward ground contact (indices 6 and 7)
    contact_bonus = 0.0
    if len(n) >= 8:
        contact_bonus = n[6] + n[7]
    
    # Component 5: Stability signal - penalize extreme state values
    # Focus on angular components (indices 4 and 5) which are critical for stability
    stability_penalty = 0.0
    if len(n) >= 6:
        stability_penalty = abs(n[4]) + abs(n[5])
    
    # Stage weights based on training_progress
    # Early stage: focus on transition progress and smoothness
    # Late stage: focus on contact and stability
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Combine components with stage weights
    reward = 0.0
    
    # Early stage: reward movement toward zero, penalize large changes
    reward += early_weight * 0.5 * transition_progress
    reward -= early_weight * 0.02 * smoothness_penalty
    
    # Late stage: reward ground contact, penalize instability and action cost
    reward += late_weight * 3.0 * contact_bonus
    reward -= late_weight * 0.5 * stability_penalty
    reward -= late_weight * 0.2 * action_cost
    
    # Small constant to encourage exploration
    reward += 0.01
    
    return reward