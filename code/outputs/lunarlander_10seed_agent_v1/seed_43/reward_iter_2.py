def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # 1. Directional progress toward zero (lower absolute values = better)
    # Negative means moving toward zero (good), positive means moving away (bad)
    progress_toward_zero = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # 2. Smoothness penalty - penalize large changes (jerky movement)
    # Use L1 norm for robustness
    smoothness = -sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # 3. Stability signal - penalize angle and angular velocity
    # obs[4] = angle, obs[5] = angular velocity
    angle_penalty = -abs(o[4]) * 0.5 - abs(o[5]) * 0.3
    
    # 4. Contact bonus - reward ground contact (legs touching ground)
    # obs[6] and obs[7] are leg contact indicators
    contact_bonus = (o[6] + o[7]) * 0.5
    
    # 5. Survival bonus - reward staying alive
    survival_bonus = 0.1
    
    # 6. Action penalty for discrete actions
    # action is 0-3, penalize non-zero actions
    action_penalty = -0.05 if action > 0 else 0.0
    
    # Stage-based weights
    # Early stage: explore, learn to move toward zero
    # Middle stage: balance progress and stability
    # Late stage: focus on stability and landing
    early_weight = max(0.0, 1.0 - 2.0 * training_progress)
    mid_weight = 1.0 - abs(2.0 * training_progress - 1.0)
    late_weight = max(0.0, 2.0 * training_progress - 1.0)
    
    # Combine components with stage weights
    reward = (
        early_weight * progress_toward_zero * 0.3 +  # Explore: learn to move toward zero
        early_weight * smoothness * 0.1 +            # Mild smoothness penalty
        mid_weight * progress_toward_zero * 0.4 +    # Mid: focus on progress
        mid_weight * angle_penalty * 0.2 +           # Mid: start caring about angle
        late_weight * angle_penalty * 0.4 +          # Late: strong stability
        late_weight * contact_bonus * 0.3 +          # Late: reward landing
        survival_bonus +                             # Always reward survival
        action_penalty                               # Penalize unnecessary actions
    )
    
    # Scale to reasonable range
    reward = reward * 0.2
    
    return reward