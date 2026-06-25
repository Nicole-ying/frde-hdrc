def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Stage weights based on training progress
    early_stage = 1.0 - training_progress
    late_stage = training_progress
    
    # Count elements for iteration
    count = min(len(obs), len(next_obs))
    
    # Component 1: Transition progress - reward movement toward center/zero
    # This is the only component that consistently correlated with better scores
    transition_progress = 0.0
    for i in range(count):
        transition_progress += abs(obs[i]) - abs(next_obs[i])
    
    # Component 2: Action cost - penalize extreme actions (away from neutral action=2)
    # Keep this but reduce its influence - it was slightly harmful in iterations 2-3
    action_penalty = -0.005 * abs(action - 2)
    
    # Component 3: Survival bonus - small constant to encourage longer episodes
    survival_bonus = 0.05
    
    # Component 4: Stability signal - penalize extreme observation values
    # This was too aggressive in previous iterations, reduce its impact
    stability_penalty = 0.0
    for i in range(count):
        stability_penalty -= 0.02 * abs(next_obs[i])
    
    # Component 5: Velocity change signal - reward reducing absolute values
    # This was redundant with transition_progress, remove it to avoid double-counting
    
    # Combine components with stage weights - simplify to focus on what works
    # Early stage: focus on transition progress (learning to move toward center)
    # Late stage: add more stability and action cost
    early_weight = early_stage * 0.8 + late_stage * 0.2
    late_weight = late_stage * 0.8 + early_stage * 0.2
    
    reward = (
        early_weight * transition_progress * 0.6 +
        late_weight * stability_penalty * 0.15 +
        action_penalty * 0.1 +
        survival_bonus * 0.15
    )
    
    return float(reward)