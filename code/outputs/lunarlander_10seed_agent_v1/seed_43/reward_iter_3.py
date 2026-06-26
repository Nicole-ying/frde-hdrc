def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # 1. Velocity reduction signal - penalize high velocity (want to land softly)
    # obs[2] = x velocity, obs[3] = y velocity
    vel_penalty = -(abs(o[2]) + abs(o[3])) * 0.3
    
    # 2. Height reduction signal - reward getting closer to ground
    # obs[1] = y position (normalized), negative is below center, positive is above
    height_penalty = -abs(o[1]) * 0.5
    
    # 3. Stability signal - penalize angle and angular velocity
    # obs[4] = angle, obs[5] = angular velocity
    angle_penalty = -abs(o[4]) * 0.5 - abs(o[5]) * 0.3
    
    # 4. Contact bonus - reward ground contact (legs touching ground)
    # obs[6] and obs[7] are leg contact indicators
    contact_bonus = (o[6] + o[7]) * 0.5
    
    # 5. Survival bonus - reward staying alive
    survival_bonus = 0.2
    
    # 6. Action smoothness - penalize large action changes
    # action is discrete 0-3, penalize main engine (2) and side engines (1,3)
    action_penalty = -0.02 * action  # action is 0-3
    
    # Stage-based weights
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage weights
    reward = (
        early_weight * height_penalty * 0.2 +          # Early: learn to descend
        early_weight * vel_penalty * 0.1 +             # Early: mild velocity penalty
        mid_weight * height_penalty * 0.3 +            # Mid: focus on height
        mid_weight * angle_penalty * 0.2 +             # Mid: start caring about angle
        mid_weight * vel_penalty * 0.2 +               # Mid: velocity matters
        late_weight * angle_penalty * 0.4 +            # Late: strong stability
        late_weight * contact_bonus * 0.5 +            # Late: reward landing
        late_weight * vel_penalty * 0.3 +              # Late: soft landing
        survival_bonus +                               # Always reward survival
        action_penalty                                 # Penalize unnecessary actions
    )
    
    # Scale to reasonable range
    reward = reward * 0.15
    
    return reward