def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Movement towards center - sum of absolute position changes
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Feature 2: Smoothness - squared difference in velocities
    vel_diff = sum((n[i] - o[i]) ** 2 for i in range(2, 4))
    
    # Feature 3: Angular stability - change in angle and angular velocity
    angle_change = abs(n[4]) - abs(o[4])
    ang_vel_change = abs(n[5]) - abs(o[5])
    
    # Feature 4: Ground contact bonus
    ground_contact = n[6] + n[7]
    
    # Feature 5: Action cost (penalize large actions)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 0.5
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3
    
    # Stage-based weights that evolve with training progress
    # Early stage: focus on exploration and basic movement
    # Middle stage: balance stability and movement
    # Late stage: refine control and precision
    
    # Sigmoid-like transition functions
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Component rewards
    # Position reward: encourage moving towards center
    pos_reward = pos_change * 0.5
    
    # Velocity smoothness: penalize large velocity changes
    smoothness_reward = -vel_diff * 0.1
    
    # Angular reward: encourage stable angle
    angle_reward = angle_change * 0.3 + ang_vel_change * 0.2
    
    # Ground contact reward: encourage landing
    contact_reward = ground_contact * 1.0
    
    # Action penalty: discourage unnecessary engine use
    action_penalty = -action_cost * 0.2
    
    # Combine with stage weights
    early_component = (pos_reward + smoothness_reward) * early_weight
    mid_component = (angle_reward + contact_reward) * mid_weight
    late_component = (contact_reward * 2.0 - action_penalty) * late_weight
    
    # Base reward for exploration
    base_reward = 0.1
    
    # Combine all components
    total_reward = base_reward + early_component + mid_component + late_component
    
    return float(total_reward)