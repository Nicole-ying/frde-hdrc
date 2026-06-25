from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from .feedback import EpisodeSummary, TrainingStats, summarize_episodes
from .reward import RewardProgram
from .wrappers import CustomRewardWrapper


@dataclass
class TrainResult:
    model: Any
    stats: TrainingStats
    interrupted: bool = False
    error_message: str = ""


class ProgressTracker:
    def __init__(self, total_timesteps: int) -> None:
        self.total_timesteps = max(1, total_timesteps)
        self.current_timesteps = 0

    def update(self, timesteps: int) -> None:
        self.current_timesteps = max(0, timesteps)

    def progress(self) -> float:
        return min(1.0, self.current_timesteps / self.total_timesteps)


def describe_space(space: Any) -> str:
    return repr(space)


def make_env(env_name: str, seed: int) -> Any:
    try:
        import gymnasium as gym
    except ImportError as exc:
        raise RuntimeError("Training requires gymnasium. Install requirements.txt first.") from exc

    env = gym.make(env_name)
    env.reset(seed=seed)
    return env


def train_agent(
    env_name: str,
    reward_program: Optional[RewardProgram],
    total_timesteps: int,
    eval_episodes: int,
    target_score: float,
    seed: int,
    training_algorithm: str = "ppo",
    progress_provider: Optional[Callable[[], float]] = None,
    ppo_params: Optional[Dict[str, Any]] = None,
) -> TrainResult:
    try:
        from stable_baselines3 import PPO
        from stable_baselines3 import DQN
        from stable_baselines3.common.callbacks import BaseCallback
        from stable_baselines3.common.vec_env import DummyVecEnv
    except ImportError as exc:
        raise RuntimeError("Training requires stable-baselines3. Install requirements.txt first.") from exc

    tracker = ProgressTracker(total_timesteps)

    class _ProgressCallback(BaseCallback):
        def _on_step(self) -> bool:
            tracker.update(self.num_timesteps)
            return True

    env = make_env(env_name, seed)
    train_env = env
    model = None
    # 🆕 从 ppo_params 中提取 n_envs（不传给 PPO 构造函数）
    n_envs = 1
    ppo_kwargs = dict(ppo_params) if ppo_params else {}
    if "n_envs" in ppo_kwargs:
        n_envs = max(1, int(ppo_kwargs.pop("n_envs")))
    try:
        if reward_program is not None:
            provider = progress_provider or tracker.progress

            def _make_env(env_seed: int = seed):
                """Create a fresh env + CustomRewardWrapper (each copy needs its own state)."""
                e = make_env(env_name, env_seed)
                return CustomRewardWrapper(e, reward_program, provider).unwrap()

            if n_envs > 1:
                # 每个并行环境用独立 seed + 独立 CustomRewardWrapper 实例
                train_env = DummyVecEnv(
                    [lambda s=seed + i: _make_env(s) for i in range(n_envs)]
                )
            else:
                train_env = _make_env(seed)

        model = _build_model(training_algorithm, train_env, seed, PPO, DQN, ppo_kwargs)
        model.learn(total_timesteps=total_timesteps, callback=_ProgressCallback())
        episodes = evaluate_model(model, env_name, eval_episodes, target_score, seed + 10_000)
        stats = summarize_episodes(episodes, target_score)
        if reward_program is not None:
            stats.reward_error_count = reward_program.error_count
            stats.reward_last_error = reward_program.last_error or ""
        return TrainResult(model=model, stats=stats)
    except Exception as exc:
        reward_error_count = reward_program.error_count if reward_program is not None else 0
        reward_last_error = reward_program.last_error if reward_program is not None else ""
        stats = TrainingStats(
            mean_eval_score=0.0,
            success_rate=0.0,
            mean_episode_length=0.0,
            trend="interrupted",
            converged=False,
            failure_mode="training interrupted before evaluation",
            interrupted=True,
            error_message=f"{type(exc).__name__}: {exc}",
            reward_error_count=reward_error_count,
            reward_last_error=reward_last_error or "",
        )
        return TrainResult(model=model, stats=stats, interrupted=True, error_message=stats.error_message)
    finally:
        train_env.close()


def _build_model(
    training_algorithm: str,
    train_env: Any,
    seed: int,
    ppo_cls: Any,
    dqn_cls: Any,
    ppo_params: Optional[Dict[str, Any]] = None,
) -> Any:
    algorithm = training_algorithm.lower()
    if algorithm == "dqn":
        return dqn_cls(
            "MlpPolicy",
            train_env,
            learning_rate=4e-3,
            buffer_size=50_000,
            learning_starts=1_000,
            batch_size=128,
            gamma=0.99,
            train_freq=4,
            target_update_interval=1_000,
            exploration_fraction=0.25,
            exploration_final_eps=0.05,
            verbose=0,
            seed=seed,
            device="cpu",
        )
    if algorithm == "ppo":
        params = {
            "learning_rate": 2.5e-4,
            "n_steps": 1024,
            "batch_size": 64,
            "n_epochs": 4,
            "gamma": 0.999,
            "gae_lambda": 0.98,
            "clip_range": 0.2,
            "ent_coef": 0.01,
            "vf_coef": 0.5,
            "max_grad_norm": 0.5,
            "verbose": 0,
            "device": "cpu",
        }
        if ppo_params:
            params.update(ppo_params)
        return ppo_cls(
            "MlpPolicy",
            train_env,
            seed=seed,
            **params,
        )
    raise ValueError(f"Unsupported training_algorithm: {training_algorithm}")


def evaluate_model(
    model: Any,
    env_name: str,
    episodes: int,
    target_score: float,
    seed: int,
) -> List[EpisodeSummary]:
    env = make_env(env_name, seed)
    summaries: List[EpisodeSummary] = []
    for episode_id in range(episodes):
        obs, _info = env.reset(seed=seed + episode_id)
        done = False
        total_reward = 0.0
        length = 0
        start_x = _extract_x(obs)
        final_x = start_x
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, _info = env.step(action)
            total_reward += float(reward)
            length += 1
            done = bool(terminated or truncated)
            final_x = _extract_x(obs)
        summaries.append(
            EpisodeSummary(
                reward=total_reward,
                length=length,
                success=total_reward >= target_score,
                start_x=start_x,
                final_x=final_x,
            )
        )
    env.close()
    return summaries


def inspect_env(env_name: str, seed: int) -> Tuple[str, str]:
    env = make_env(env_name, seed)
    observation = describe_space(env.observation_space)
    action = describe_space(env.action_space)
    env.close()
    return observation, action


def known_env_description(env_name: str) -> str:
    """Return optional human-provided observation semantics.

    The autonomous FDRE-HRDC path intentionally leaves this blank so reward search is
    driven by black-box interfaces, scalar environment rewards, and training feedback.
    """
    return ""


def _extract_x(obs: Any) -> float:
    try:
        return float(obs[0])
    except Exception:
        return 0.0
