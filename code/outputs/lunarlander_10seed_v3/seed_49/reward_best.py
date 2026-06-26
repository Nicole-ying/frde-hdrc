def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement towards stability - penalize large changes
    delta = n - o
    movement_cost = sum(delta * delta) * 0.01
    
    # Component 2: Encourage convergence - reward when absolute values decrease
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    convergence_bonus = abs_diff * 0.1
    
    # Component 3: Action penalty - discourage excessive action usage
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.25
    
    # Component 4: Contact bonus from info
    contact_bonus = 0.0
    if info and 'ground_contact' in info:
        contact_bonus = info['ground_contact'] * 0.5
    
    # Stage-based weighting
    stage1_weight = max(0.0, 1.0 - training_progress * 2.0)  # Early: focus on movement
    stage2_weight = min(1.0, training_progress * 2.0)  # Late: focus on convergence
    
    # Combine components
    reward = (
        stage1_weight * (-movement_cost + action_cost) +
        stage2_weight * (convergence_bonus + contact_bonus)
    )
    
    return float(reward)