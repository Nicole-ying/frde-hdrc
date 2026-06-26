def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Sum of absolute differences (movement magnitude)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (energy-like signal)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action cost (penalize action usage)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0
    elif action == 1 or action == 3:
        action_cost = 0.1
    elif action == 2:
        action_cost = 0.2
    
    # Feature 4: Ground contact signal from info (if available)
    ground_contact = 0.0
    if 'ground_contact' in info:
        ground_contact = float(info['ground_contact'])
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on exploration and movement
    # Middle stage: balance movement and stability
    # Late stage: emphasize precision and efficiency
    
    # Sigmoid-like transition based on training_progress
    stage_factor = 1.0 / (1.0 + 2.71828 ** (-10.0 * (training_progress - 0.5)))
    
    # Component weights that shift across stages
    w_abs_diff = 0.5 * (1.0 - stage_factor)  # Decreases over time
    w_sq_change = 0.3 * (1.0 - 0.5 * stage_factor)  # Moderate decrease
    w_action_cost = -0.1 * (1.0 + stage_factor)  # Increases penalty over time
    w_ground = 0.2 * stage_factor  # Increases over time
    
    # Compute reward components
    reward_abs_diff = w_abs_diff * abs_diff
    reward_sq_change = w_sq_change * sq_change
    reward_action_cost = w_action_cost * action_cost
    reward_ground = w_ground * ground_contact
    
    # Combine components
    reward = reward_abs_diff + reward_sq_change + reward_action_cost + reward_ground
    
    # Ensure numerical stability by keeping values reasonable
    # The training framework will clip the final reward
    
    return float(reward)