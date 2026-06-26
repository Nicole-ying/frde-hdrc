def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features - dimension agnostic
    # Feature 1: Sum of absolute differences (movement magnitude)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Sum of squared differences (smoothness/change)
    sq_diff = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (encourage efficient actions)
    # Action is discrete (0-3), normalize to small cost
    action_cost = 0.01 * action
    
    # Feature 4: Contact bonus from info (if available)
    contact_bonus = 0.0
    if info and isinstance(info, dict):
        # Generic contact signal from info
        contact_bonus = info.get('contact', 0.0) * 0.1
    
    # Stage weights based on training_progress (0.0 to 1.0)
    # Early stage: focus on exploration and movement
    # Middle stage: balance movement and stability
    # Late stage: focus on precision and efficiency
    
    # Weight for movement reward (abs_diff)
    w_movement = 0.5 * (1.0 - training_progress * 0.3)
    
    # Weight for smoothness (negative sq_diff - penalize large changes)
    w_smooth = -0.2 * (0.5 + training_progress * 0.5)
    
    # Weight for action efficiency (negative cost)
    w_action = -0.1 * (0.3 + training_progress * 0.7)
    
    # Weight for contact bonus
    w_contact = 0.3 * (training_progress * 0.8)
    
    # Combine components
    reward = (
        w_movement * abs_diff +
        w_smooth * sq_diff +
        w_action * action_cost +
        w_contact * contact_bonus
    )
    
    # Ensure numerical stability by bounding extreme values
    # Use tanh-like scaling via division by large number
    reward = reward / (1.0 + abs(reward) * 0.01)
    
    return float(reward)