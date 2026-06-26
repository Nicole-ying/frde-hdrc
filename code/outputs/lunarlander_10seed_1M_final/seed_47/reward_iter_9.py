def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Create generic arrays for dimension-agnostic processing
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability - penalize large changes in absolute values
    # This encourages smooth transitions
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    movement_reward = abs_diff * 0.1
    
    # Component 2: Smoothness - penalize large jumps in state
    # Smaller squared differences indicate smoother transitions
    squared_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    smoothness_penalty = -squared_diff * 0.05
    
    # Component 3: Action cost - penalize frequent actions
    # Small action cost encourages efficiency
    action_cost = -0.01 * float(action)
    
    # Component 4: Contact bonus - reward ground contact if present in info
    # Generic check for any contact or success signals
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = 0.1 * float(info['contact'])
    
    # Stage-based weights that evolve with training_progress
    # Early training: focus on smoothness and movement
    # Late training: focus on contact and efficiency
    stage1_weight = max(0.0, 1.0 - 2.0 * training_progress)  # Decreases from 1 to 0
    stage2_weight = min(1.0, 2.0 * training_progress)  # Increases from 0 to 1
    
    # Combine components with stage weights
    reward = (
        stage1_weight * (movement_reward + smoothness_penalty) +
        stage2_weight * (action_cost + contact_bonus)
    )
    
    return reward