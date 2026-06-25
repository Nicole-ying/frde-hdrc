def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    # Extract observation arrays
    o = obs
    n = next_obs

    # ========== SIGNAL CATEGORIES ==========

    # 1) TRANSITION PROGRESS: directional movement toward desirable states
    #    Use position (indices 0,1) and velocity (indices 2,3) changes toward zero
    pos_progress = sum(abs(o[i]) - abs(n[i]) for i in range(2))  # x,y position
    vel_progress = sum(abs(o[i]) - abs(n[i]) for i in range(2, 4))  # vx,vy velocity

    # 2) STABILITY: reward angle and angular velocity improvement (directional toward zero)
    angle_progress = abs(o[4]) - abs(n[4])  # positive when angle decreases
    ang_vel_progress = abs(o[5]) - abs(n[5])  # positive when angular velocity decreases

    # 3) CONTACT/LANDING SIGNAL: reward gaining ground contact
    contact_gain = sum(n[6:8]) - sum(o[6:8])  # positive when new legs touch down

    # 4) SMOOTHNESS: penalize large changes in velocity (jerk)
    vel_jerk = sum((n[i] - o[i]) ** 2 for i in range(2, 4))

    # 5) ACTION COST: penalize engine usage
    #    For discrete actions: 0=do nothing, 1=left, 2=main, 3=right
    #    Penalize main engine (action==2) and side engines (action==1 or 3)
    action_penalty = 0.0
    if action == 2:  # main engine
        action_penalty = 1.0
    elif action in [1, 3]:  # side engines
        action_penalty = 0.5

    # 6) SURVIVAL BONUS: reward staying alive longer
    survival_bonus = 0.05  # small constant per step

    # ========== STAGE WEIGHTS ==========
    # Early: explore, learn to move and stabilize
    # Mid: refine landing, reduce action cost
    # Late: maximize contact, minimize all penalties

    # Reduce all weights significantly to prevent large negative rewards
    # that cause early termination
    w_pos = 0.3 + 0.3 * training_progress  # position progress always important
    w_vel = 0.2 + 0.2 * training_progress  # velocity progress grows
    w_angle = 0.2 * (1.0 + 0.5 * training_progress)  # angle progress becomes more important over time
    w_contact = 0.5 * training_progress  # contact reward ramps up
    w_jerk = 0.05 * (1.0 - 0.5 * training_progress)  # smoothness matters more early
    w_action = 0.05 * (1.0 + 0.5 * training_progress)  # action cost increases over time
    w_survival = 0.5 * (1.0 - 0.5 * training_progress)  # survival bonus decreases over time

    # ========== COMBINE ==========
    reward = (
        w_pos * pos_progress
        + w_vel * vel_progress
        + w_angle * (angle_progress + ang_vel_progress)
        + w_contact * contact_gain
        - w_jerk * vel_jerk
        - w_action * action_penalty
        + w_survival * survival_bonus
    )

    return reward