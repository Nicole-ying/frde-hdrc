def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement magnitude - encourage reducing absolute position/velocity
    movement_reduction = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Feature 2: Smoothness - penalize large changes in state
    state_change_penalty = -sum((n[i] - o[i]) ** 2 for i in range(8))
    
    # Feature 3: Ground contact bonus - encourage landing (leg contact signals)
    ground_contact = sum(n[6:8])  # Sum of leg contact indicators
    
    # Feature 4: Stability - penalize angular velocity and angle
    angular_penalty = -abs(n[4]) - abs(n[5])
    
    # Feature 5: Action cost - small penalty for taking actions
    action_cost = -0.01 * action
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on movement and exploration
    # Middle stage: balance smoothness and contact
    # Late stage: emphasize stability and landing
    
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)  # Decreases from 1 to 0
    stage2_weight = 1.0 - abs(2.0 * training_progress - 1.0)  # Peaks at 0.5
    stage3_weight = max(0.0, 2.0 * training_progress - 1.0)  # Increases from 0 to 1
    
    # Combine components with stage weights
    reward = (
        stage1_weight * (0.5 * movement_reduction + 0.1 * state_change_penalty) +
        stage2_weight * (0.3 * state_change_penalty + 0.5 * ground_contact) +
        stage3_weight * (0.7 * ground_contact + 0.3 * angular_penalty) +
        action_cost
    )
    
    return reward