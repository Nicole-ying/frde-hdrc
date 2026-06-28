def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    total = 0.0
    count = min(len(obs), len(next_obs))
    for i in range(count):
        total += abs(float(obs[i])) - abs(float(next_obs[i]))
    return float(total)
