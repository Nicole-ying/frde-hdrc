def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features
    # Feature 1: Movement towards center - sum of absolute position changes
    pos_change = sum(abs(o[i]) - abs(n[i]) for i in range(4))
    
    # Feature 2: Velocity change - squared differences
    vel_change = sum((n[i] - o[i]) ** 2 for i in range(2, 4))
    
    # Feature 3: Angular stability - angle and angular velocity changes
    angle_change = abs(n[4]) - abs(o[4])
    ang_vel_change = abs(n[5]) - abs(o[5])
    
    # Feature 4: Ground contact bonus
    ground_contact = sum(n[6:8]) - sum(o[6:8])
    
    # Feature 5: Action penalty (small cost to encourage efficient actions)
    action_cost = 0.01 * action
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on reaching center and stability
    # Middle stage: balance movement and stability
    # Late stage: fine-tune landing with ground contact
    
    # Sigmoid-like transition based on training_progress
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Component rewards with stage-adaptive weights
    reward = 0.0
    
    # Position reward (early focus)
    reward += early_weight * 0.5 * pos_change
    
    # Velocity penalty (constant throughout)
    reward += -0.1 * vel_change
    
    # Angle stability (mid focus)
    reward += mid_weight * 0.3 * (-angle_change - 0.1 * ang_vel_change)
    
    # Ground contact bonus (late focus)
    reward += late_weight * 2.0 * ground_contact
    
    # Small action penalty (always present)
    reward += -action_cost
    
    return reward