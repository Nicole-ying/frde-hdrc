def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement magnitude - encourage reducing absolute values (stabilization)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Smoothness - penalize large changes (encourage gentle transitions)
    smoothness = -sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty - discourage excessive action usage
    action_cost = -0.01 * abs(action - 1.5)  # Center action penalty
    
    # Stage-based weights
    # Early stage: focus on stabilization and smoothness
    # Late stage: maintain stability with less aggressive smoothing
    w1 = 1.0 - 0.5 * training_progress  # abs_diff weight decreases
    w2 = 0.5 + 0.5 * training_progress   # smoothness weight increases
    w3 = 0.1 * (1.0 - training_progress) # action cost weight decreases
    
    # Combine components
    reward = w1 * abs_diff + w2 * smoothness + w3 * action_cost
    
    return reward