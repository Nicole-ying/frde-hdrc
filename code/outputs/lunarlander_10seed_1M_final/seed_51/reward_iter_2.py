def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observations as generic arrays
    o = obs
    n = next_obs

    # --- Signal inventory ---
    # 1. Convergence toward zero: directional signal that rewards moving states toward origin
    convergence = sum(abs(o[i]) - abs(n[i]) for i in range(len(o)))

    # 2. Smoothness penalty: penalizes large jumps in state (encourages stable transitions)
    abs_change = sum(abs(n[i] - o[i]) for i in range(len(o)))

    # 3. Action cost: penalizes large actions (encourages efficient control)
    action_cost = abs(action)

    # 4. Survival bonus: reward for staying alive (counteracts early termination)
    survival_bonus = 1.0

    # --- Stage weights ---
    # Early: explore with moderate convergence, low penalties, small survival bonus
    # Middle: increase convergence pressure, moderate smoothness, moderate survival
    # Late: strong convergence, high smoothness, low action cost, high survival
    if training_progress < 0.3:
        w_converge = 1.0
        w_smooth = 0.2
        w_action = 0.05
        w_survival = 0.3
    elif training_progress < 0.7:
        w_converge = 2.0
        w_smooth = 0.4
        w_action = 0.03
        w_survival = 0.5
    else:
        w_converge = 3.0
        w_smooth = 0.7
        w_action = 0.01
        w_survival = 0.8

    # --- Combine components ---
    reward = (
        w_converge * convergence
        - w_smooth * abs_change
        - w_action * action_cost
        + w_survival * survival_bonus
    )

    return reward