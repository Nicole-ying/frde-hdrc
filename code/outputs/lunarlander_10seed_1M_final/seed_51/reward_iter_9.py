def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features - dimension agnostic
    # Feature 1: Movement magnitude (change in absolute values)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared distance moved
    squared_dist = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (small cost for taking actions)
    action_cost = 0.01 * action
    
    # Feature 4: Stability signal - penalize large changes in observations
    stability = -sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on movement and exploration
    # Mid training: balance movement with stability
    # Late training: emphasize stability and precision
    
    # Sigmoid-like transition based on training_progress
    early_weight = max(0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage weights
    reward = (
        early_weight * (0.5 * abs_change - 0.1 * squared_dist - action_cost) +
        mid_weight * (0.3 * stability - 0.05 * squared_dist) +
        late_weight * (0.8 * stability - 0.2 * squared_dist)
    )
    
    return reward