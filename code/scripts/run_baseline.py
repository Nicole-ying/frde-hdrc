"""Run PPO baseline with original environment reward (no LLM reward wrapper)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from llm_reward_evolver.config import ExperimentConfig
from llm_reward_evolver.trainer import train_agent

config = ExperimentConfig().with_overrides(
    env_name="LunarLander-v3",
    total_timesteps=2_000_000,
    eval_episodes=100,
    target_score=999,
    ppo_n_envs=4,
    ppo_verbose=1,
)

output_dir = ROOT / "outputs/baseline_2M"
output_dir.mkdir(parents=True, exist_ok=True)

scores = []
for seed in [42, 43, 44]:
    result = train_agent(
        config.env_name,
        reward_program=None,        # None = 使用环境原始奖励
        total_timesteps=config.total_timesteps,
        eval_episodes=config.eval_episodes,
        target_score=config.target_score,
        seed=seed,
        training_algorithm="ppo",
        ppo_params=config.ppo_kwargs(),
    )
    s = result.stats
    scores.append(s.mean_eval_score)

    # 保存模型
    if result.model:
        result.model.save(str(output_dir / f"baseline_seed{seed}.zip"))

    print(f"[baseline] seed={seed}: score={s.mean_eval_score:.1f}, "
          f"success_rate={s.success_rate:.3f}, len={s.mean_episode_length:.0f}")

mean = sum(scores) / len(scores)
print(f"\nBaseline 2M: mean={mean:.1f} across {len(scores)} seeds")
print(f"Scores: {scores}")
print(f"Use mean={mean:.1f} as the 'solved' threshold for FDRE-HRDC experiments.")
