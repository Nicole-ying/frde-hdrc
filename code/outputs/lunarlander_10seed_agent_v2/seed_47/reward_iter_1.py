def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs
    
    # Generic transition features (dimension-agnostic)
    # Feature 1: Directional movement - encourage moving toward zero-centered states
    # Using abs(o) - abs(n) which rewards moving toward zero
    directional_progress = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))
    
    # Feature 2: Velocity change penalty - penalize large velocity changes (smoothness)
    # Use absolute difference instead of squared to be more gentle
    smoothness = -sum(abs(n[i] - o[i]) for i in range(len(o)))
    
    # Feature 3: Action penalty - small cost for using engines
    action_cost = -0.01 if action != 0 else 0.0
    
    # Feature 4: Stability signal - penalize extreme values in any dimension
    # This encourages staying in a stable region
    stability = -sum(abs(n[i]) for i in range(len(o)))
    
    # Stage-based weighting
    if training_progress < 0.3:
        # Early exploration: focus on movement and exploration
        w_directional = 1.0
        w_smoothness = 0.2
        w_action = -0.01
        w_stability = 0.1
    elif training_progress < 0.7:
        # Middle refinement: balance progress with stability
        w_directional = 0.8
        w_smoothness = 0.5
        w_action = -0.02
        w_stability = 0.3
    else:
        # Late precision: prioritize stability and smoothness
        w_directional = 0.3
        w_smoothness = 1.0
        w_action = -0.03
        w_stability = 0.8
    
    # Combine components
    reward = (w_directional * directional_progress +
              w_smoothness * smoothness +
              w_action * action_cost +
              w_stability * stability)
    
    # Small constant to encourage exploration
    reward = reward + 0.1
    
    return float(reward)