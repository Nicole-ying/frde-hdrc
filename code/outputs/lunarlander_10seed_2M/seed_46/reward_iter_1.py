def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract generic arrays
    o = obs
    n = next_obs
    
    # Component 1: Transition progress - reward movement toward zero (directional)
    # This measures if the agent is moving toward a stable state (smaller absolute values)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness penalty - penalize large jumps in state
    # Use L1 norm for robustness
    delta = n - o
    jerkiness = sum(abs(d) for d in delta)
    
    # Component 3: Action cost - penalize engine usage
    # For discrete actions, action is a scalar integer (0,1,2,3)
    # Action 2 is main engine, 1 and 3 are side engines, 0 is no engine
    action_penalty = 0.0
    if action == 2:  # Main engine
        action_penalty = 1.0
    elif action in [1, 3]:  # Side engines
        action_penalty = 0.5
    
    # Component 4: Stability signal - penalize angular velocity and angle
    # obs[4] is angle, obs[5] is angular velocity (from environment step source)
    angle_penalty = abs(o[4]) + abs(o[5])  # Penalize being tilted or spinning
    
    # Component 5: Velocity penalty - discourage high speed (crash risk)
    # obs[2] is x velocity, obs[3] is y velocity
    speed = abs(o[2]) + abs(o[3])
    velocity_penalty = speed
    
    # Component 6: Height-based reward - reward being at reasonable height
    # obs[1] is normalized y position (0 is helipad level)
    height = abs(o[1])  # Penalize being far from helipad height
    
    # Component 7: Contact bonus from info
    contact_bonus = 0.0
    if 'contact' in info:
        contact_bonus = info['contact']  # 0 or 1 typically
    
    # Stage-based weights
    early_weight = 1.0 - training_progress
    late_weight = training_progress
    
    # Combine components with stage-adaptive weights
    # Early: focus on exploration and learning to move
    # Late: focus on stability and landing
    reward = (
        # Transition progress - reward moving toward zero (directional)
        0.3 * abs_diff * (0.5 + 0.5 * early_weight) +
        
        # Smoothness penalty - small penalty for jerky movement
        -0.05 * jerkiness * (0.3 + 0.7 * late_weight) +
        
        # Action cost - discourage unnecessary engine use
        -0.1 * action_penalty * (0.5 + 0.5 * late_weight) +
        
        # Stability - strongly penalize angle and angular velocity
        -0.2 * angle_penalty * (0.3 + 0.7 * late_weight) +
        
        # Velocity penalty - discourage high speed
        -0.1 * velocity_penalty * (0.5 + 0.5 * late_weight) +
        
        # Height penalty - encourage being near helipad
        -0.1 * height * (0.3 + 0.7 * late_weight) +
        
        # Contact bonus - reward ground contact (landing)
        0.5 * contact_bonus * late_weight
    )
    
    return reward