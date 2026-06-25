def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # ===== SIGNAL CATEGORIES =====

    # 1. TRANSITION SIGNAL: movement toward desirable states (directional)
    # Encourage moving toward zero for position/velocity/angle (typical goal)
    abs_change = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # 2. SMOOTHNESS SIGNAL: penalize large jumps between states
    # Use absolute change to avoid over-penalizing
    smoothness = -sum(abs(n[i] - o[i]) for i in range(len(o)))

    # 3. STABILITY SIGNAL: penalize angular velocity and angle (prevent tumbling)
    # obs[4] = angle, obs[5] = angular velocity
    angle_penalty = 0.0
    angvel_penalty = 0.0
    if len(o) >= 6:
        # Penalize large angles (away from upright)
        angle_penalty = -abs(o[4]) * 0.5  # directional: reduce angle magnitude
        # Penalize angular velocity (spinning)
        angvel_penalty = -abs(o[5]) * 0.3

    # 4. VELOCITY SIGNAL: penalize high velocity (soft landing)
    velocity_penalty = 0.0
    if len(o) >= 4:
        # Penalize horizontal and vertical velocity magnitude
        velocity_penalty = -((o[2]**2 + o[3]**2) ** 0.5) * 0.2

    # 5. CONTACT/LANDING SIGNAL: reward ground contact
    ground_contact = 0.0
    if len(o) >= 8:
        ground_contact = o[6] + o[7]  # leg contact indicators

    # 6. ACTION COST: penalize engine usage (efficiency)
    # For discrete actions, penalize main engine (action=2) and side engines (1,3)
    action_cost = 0.0
    if action == 2:  # main engine
        action_cost = -0.5
    elif action in [1, 3]:  # side engines
        action_cost = -0.2

    # ===== STAGE WEIGHTS =====
    # Early: explore, learn to stabilize and move
    # Middle: refine smoothness and landing
    # Late: maximize contact and minimize velocity

    # Sigmoid-like progression with three stages
    early_weight = 1.0 - training_progress  # decreases from 1 to 0
    middle_weight = 4.0 * training_progress * (1.0 - training_progress)  # peaks at 0.5
    late_weight = training_progress  # increases from 0 to 1

    # ===== COMBINE COMPONENTS =====
    reward = (
        # Early stage: focus on stability and movement toward good states
        early_weight * (abs_change * 0.2 + angle_penalty + angvel_penalty) +
        # Middle stage: smoothness and velocity control
        middle_weight * (smoothness * 0.1 + velocity_penalty) +
        # Late stage: landing and efficiency
        late_weight * (ground_contact * 1.0 + action_cost * 0.5) +
        # Constant small survival bonus to encourage longer episodes
        0.05
    )

    return reward