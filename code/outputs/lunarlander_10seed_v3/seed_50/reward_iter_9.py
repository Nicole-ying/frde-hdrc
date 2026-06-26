def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # Component 1: Directional movement toward center (only first 4 state dimensions)
    # Positive when moving toward zero (desirable)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(min(4, len(o))))

    # Component 2: Smoothness penalty - penalize large state changes
    # Use L1 norm to reduce scale
    smoothness = sum(abs(n[i] - o[i]) for i in range(len(o)))

    # Component 3: Action penalty (small cost for taking actions)
    action_cost = 0.0
    if action == 2:  # Main engine
        action_cost = 1.0
    elif action in [1, 3]:  # Side engines
        action_cost = 0.3

    # Component 4: Ground contact bonus (from observation indices 6 and 7)
    ground_contact = 0.0
    if len(o) >= 8:
        ground_contact = n[6] + n[7]  # Use next_obs for current contact status

    # Component 5: Velocity penalty - discourage high speeds (indices 2,3 are velocities)
    vel_penalty = abs(n[2]) + abs(n[3])

    # Component 6: Angle penalty - discourage tilting (index 4 is angle)
    angle_penalty = abs(n[4])

    # Component 7: Angular velocity penalty - discourage spinning (index 5)
    ang_vel_penalty = abs(n[5])

    # Stage-based weights that evolve with training progress
    # Early stage: focus on exploration and movement
    # Middle stage: balance movement and stability
    # Late stage: prioritize precision and ground contact

    w1 = 1.0 + 0.5 * training_progress  # Movement weight increases
    w2 = 0.1 + 0.1 * training_progress  # Smoothness penalty increases slightly
    w3 = 0.1 * (1.0 - training_progress)  # Action cost decreases
    w4 = 0.5 * training_progress  # Ground contact weight increases
    w5 = 0.2 * (1.0 - 0.5 * training_progress)  # Velocity penalty decreases
    w6 = 0.3 * (1.0 - 0.5 * training_progress)  # Angle penalty decreases
    w7 = 0.1 * (1.0 - 0.5 * training_progress)  # Angular velocity penalty decreases

    # Combine components
    reward = (w1 * abs_change
              - w2 * smoothness
              - w3 * action_cost
              + w4 * ground_contact
              - w5 * vel_penalty
              - w6 * angle_penalty
              - w7 * ang_vel_penalty)

    return reward