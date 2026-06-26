def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # 1. Movement magnitude (change in absolute values)
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 2. Squared change (captures large deviations)
    squared_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # 3. Action penalty (encourage efficiency)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.2
    
    # Stage weights based on training progress
    # Early: focus on movement and exploration
    # Middle: balance movement and stability
    # Late: emphasize fine control and efficiency
    if training_progress < 0.3:
        w_movement = 1.0
        w_squared = 0.1
        w_action = 0.3
    elif training_progress < 0.7:
        w_movement = 0.5
        w_squared = 0.5
        w_action = 0.5
    else:
        w_movement = 0.2
        w_squared = 1.0
        w_action = 0.8
    
    # Combine components
    reward = (
        w_movement * movement
        - w_squared * squared_change
        - w_action * action_cost
    )
    
    return reward