def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Component 1: Directional progress toward zero-centered states
    # Rewards moving toward zero, penalizes moving away
    directional_progress = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Component 2: Smoothness - penalize large state changes
    # Uses absolute difference (gentler than squared)
    smoothness = -sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Component 3: Action cost - small penalty for using engines
    action_cost = -0.01 if action != 0 else 0.0
    
    # Component 4: Survival bonus - reward for staying alive
    survival_bonus = 0.5
    
    # Component 5: Stability signal - penalize extreme values
    # This encourages staying in a stable region
    stability = -sum(abs(n[i]) for i in range(len(o))) * 0.1
    
    # Component 6: Contact bonus - reward for ground contact
    # Use info to detect contact if available, otherwise use observation
    # Observations 6 and 7 are leg contact indicators
    contact_bonus = 0.0
    if len(o) >= 8:
        # Leg contact signals from observation
        leg1_contact = 1.0 if n[6] > 0.5 else 0.0
        leg2_contact = 1.0 if n[7] > 0.5 else 0.0
        contact_bonus = (leg1_contact + leg2_contact) * 0.5
    
    # Stage-based weighting
    if training_progress < 0.3:
        # Early exploration: focus on movement and survival
        w_directional = 1.0
        w_smoothness = 0.2
        w_action = -0.01
        w_survival = 0.5
        w_stability = 0.1
        w_contact = 0.2
    elif training_progress < 0.7:
        # Middle refinement: balance all components
        w_directional = 0.8
        w_smoothness = 0.5
        w_action = -0.02
        w_survival = 0.8
        w_stability = 0.3
        w_contact = 0.5
    else:
        # Late precision: prioritize stability and smoothness
        w_directional = 0.3
        w_smoothness = 1.0
        w_action = -0.03
        w_survival = 1.0
        w_stability = 0.5
        w_contact = 1.0
    
    # Combine components
    reward = (w_directional * directional_progress +
              w_smoothness * smoothness +
              w_action * action_cost +
              w_survival * survival_bonus +
              w_stability * stability +
              w_contact * contact_bonus)
    
    return float(reward)