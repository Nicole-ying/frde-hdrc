def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards center - encourages reducing absolute position values
    # Uses sum of absolute values reduction as a generic stabilization signal
    pos_reduction = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Component 2: Velocity damping - encourages reducing motion
    # Uses squared difference to penalize large changes in velocity-related dimensions
    vel_change = sum((n[i] - o[i]) ** 2 for i in range(2, 4))
    
    # Component 3: Angular stabilization - encourages keeping angle and angular velocity small
    angle_stability = sum(abs(o[i]) - abs(n[i]) for i in range(4, 6))
    
    # Component 4: Ground contact bonus - encourages landing (dimensions 6 and 7 are contact flags)
    contact_gain = sum(n[i] - o[i] for i in range(6, 8))
    
    # Component 5: Action cost - small penalty to discourage excessive action
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.1
    elif action in [1, 3]:  # Side engines
        action_cost = 0.05
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance all components
    # Late stage: emphasize stabilization and landing
    
    if training_progress < 0.3:
        # Early training: encourage exploration and basic control
        w1 = 1.0  # Position reduction
        w2 = 0.3  # Velocity damping
        w3 = 0.5  # Angular stabilization
        w4 = 0.2  # Contact gain
        w5 = 0.1  # Action cost
    elif training_progress < 0.7:
        # Middle training: balanced approach
        w1 = 0.8
        w2 = 0.6
        w3 = 0.7
        w4 = 0.5
        w5 = 0.2
    else:
        # Late training: focus on precision and landing
        w1 = 0.5
        w2 = 0.8
        w3 = 0.9
        w4 = 1.0
        w5 = 0.3
    
    # Combine components with stage weights
    reward = (
        w1 * pos_reduction +
        w2 * (-vel_change) +  # Negative because we want to reduce velocity changes
        w3 * angle_stability +
        w4 * contact_gain +
        w5 * (-action_cost)  # Negative because action cost is a penalty
    )
    
    return reward