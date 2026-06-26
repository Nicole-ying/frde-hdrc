def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features - dimension agnostic
    # Feature 1: Sum of absolute differences (movement magnitude)
    abs_diff = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Squared change (smoothness/energy)
    sq_change = sum((n[i] - o[i]) ** 2 for i in range(len(o)))
    
    # Feature 3: Action penalty (encourage efficient control)
    action_cost = 0.0
    if action == 0:
        action_cost = 0.0  # no action
    elif action == 1 or action == 3:
        action_cost = 0.1  # side engine
    elif action == 2:
        action_cost = 0.2  # main engine
    
    # Feature 4: Leg contact bonus (from info or observation)
    # Using last two dimensions of observation which indicate ground contact
    leg_contact = 0.0
    if len(o) >= 8:
        leg_contact = n[6] + n[7]  # sum of leg contact indicators
    
    # Feature 5: Velocity change (from observation indices 2,3 which are velocities)
    vel_change = 0.0
    if len(o) >= 4:
        vel_change = abs(n[2]) + abs(n[3]) - (abs(o[2]) + abs(o[3]))
    
    # Stage-based weights that evolve with training_progress
    # Early stage: focus on exploration and movement
    # Middle stage: balance movement and control
    # Late stage: focus on precision and stability
    
    # Sigmoid-like transition functions
    early_weight = max(0.0, 1.0 - training_progress * 2.0)
    mid_weight = 1.0 - abs(training_progress - 0.5) * 2.0
    late_weight = max(0.0, training_progress * 2.0 - 1.0)
    
    # Component rewards
    # Movement reward: encourage reducing absolute values (stabilizing)
    movement_reward = abs_diff * 0.01
    
    # Smoothness penalty: discourage large changes
    smoothness_penalty = -sq_change * 0.005
    
    # Action efficiency: penalize fuel usage
    action_penalty = -action_cost * 0.1
    
    # Contact bonus: reward leg contact (landing)
    contact_bonus = leg_contact * 0.5
    
    # Velocity reduction: reward slowing down
    velocity_reward = -vel_change * 0.02
    
    # Combine with stage weights
    reward = (
        early_weight * (movement_reward * 0.3 + action_penalty * 0.7) +
        mid_weight * (movement_reward * 0.2 + smoothness_penalty * 0.3 + velocity_reward * 0.5) +
        late_weight * (contact_bonus * 0.6 + smoothness_penalty * 0.2 + velocity_reward * 0.2)
    )
    
    return reward