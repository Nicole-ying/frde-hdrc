def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Movement magnitude change (encourages beneficial state changes)
    # Sum of absolute differences between current and next observation
    movement = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: State change magnitude (captures overall transition activity)
    state_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Component 3: Action penalty (encourage efficient actions)
    action_cost = -0.01 * action
    
    # Stage-based weights that evolve with training progress
    # Early training: focus on exploration and movement
    # Mid training: balance movement and state changes
    # Late training: refine with smaller adjustments
    
    # Weight for movement component - decreases over time
    w_movement = 0.5 * (1.0 - training_progress)
    
    # Weight for state change - increases then stabilizes
    w_state = 0.3 * (0.5 + 0.5 * training_progress)
    
    # Weight for action cost - increases to encourage efficiency
    w_action = 0.1 * (0.2 + 0.8 * training_progress)
    
    # Combine components
    reward = (w_movement * movement + 
              w_state * state_change + 
              w_action * action_cost)
    
    return reward