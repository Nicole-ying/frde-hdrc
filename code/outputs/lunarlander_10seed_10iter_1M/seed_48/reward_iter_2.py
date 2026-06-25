def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Stage weights based on training progress
    early_stage = 1.0 - training_progress
    late_stage = training_progress

    # Component 1: Transition progress - reward movement toward center/zero
    # Use absolute value reduction as a generic stability signal
    transition_progress = 0.0
    count = min(len(obs), len(next_obs))
    for i in range(count):
        transition_progress += abs(obs[i]) - abs(next_obs[i])

    # Component 2: Smoothness penalty - penalize large action changes
    # Action is a scalar integer for discrete actions
    action_penalty = -0.01 * abs(action - 2)  # Penalize away from neutral action

    # Component 3: Survival bonus - reward staying alive
    survival_bonus = 0.1

    # Component 4: Stability signal - penalize extreme observations
    stability_penalty = 0.0
    for i in range(count):
        stability_penalty -= 0.05 * abs(next_obs[i])

    # Component 5: Velocity change signal - reward reducing velocity magnitude
    # Generic signal: penalize increases in absolute observation values
    velocity_change = 0.0
    for i in range(count):
        velocity_change += (abs(obs[i]) - abs(next_obs[i])) * 0.1

    # Combine components with stage weights
    early_weight = early_stage * 0.6 + late_stage * 0.4
    late_weight = late_stage * 0.6 + early_stage * 0.4

    reward = (
        early_weight * transition_progress * 0.4 +
        late_weight * stability_penalty * 0.2 +
        action_penalty * 0.1 +
        survival_bonus * 0.15 +
        velocity_change * 0.15
    )

    return float(reward)