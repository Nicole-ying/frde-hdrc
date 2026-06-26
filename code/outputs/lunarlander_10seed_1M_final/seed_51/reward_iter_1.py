def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs

    # --- Signal inventory ---
    # 1. Convergence toward zero: directional signal that rewards moving states toward origin
    convergence = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # 2. Smoothness penalty: penalizes large jumps in state (encourages stable transitions)
    #    Using L1 change as it's more robust than L2
    abs_change = sum(abs(n[i] - o[i]) for i in range(len(o)))

    # 3. Action cost: penalizes large actions (encourages efficient control)
    action_cost = abs(action)

    # --- Stage weights ---
    # Early: explore with moderate convergence, low penalties
    # Middle: increase convergence pressure, moderate smoothness
    # Late: strong convergence, high smoothness, low action cost
    if training_progress < 0.3:
        w_converge = 1.0
        w_smooth = 0.3
        w_action = 0.05
    elif training_progress < 0.7:
        w_converge = 2.0
        w_smooth = 0.5
        w_action = 0.03
    else:
        w_converge = 3.0
        w_smooth = 0.8
        w_action = 0.01

    # --- Combine components ---
    # Convergence is positive (reward moving toward zero)
    # Smoothness is negative (penalize large changes)
    # Action cost is negative (penalize large actions)
    reward = (
        w_converge * convergence
        - w_smooth * abs_change
        - w_action * action_cost
    )

    return reward