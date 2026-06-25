from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class ExperimentConfig:
    env_name: str = "CartPole-v1"
    task_description: str = "Keep the agent alive and solve the task."
    task_description_path: str = ""
    step_source_path: str = ""
    mask_official_reward: bool = True
    method: str = "fdre_hrdc"
    llm_provider: str = "mock"
    llm_model: str = "mock-reward-coder"
    max_iterations: int = 3
    total_timesteps: int = 50_000
    eval_episodes: int = 5
    target_score: float = 475.0
    reward_clip: float = 10.0
    output_dir: str = "outputs/run"
    seed: int = 42
    patience: int = 3
    min_improvement: float = 1.0
    force_iterations_before_patience: int = 2
    reward_error_fallback: str = "original"
    reward_repair_attempts: int = 1
    num_seeds: int = 1
    feedback_mode: str = "diagnostic"
    reward_structure: str = "hrdc"
    training_algorithm: str = "ppo"
    min_original_reward_anchor: float = 0.5
    allow_original_reward_in_reward: bool = False
    restart_after_bad_iterations: int = 4
    restart_score_threshold: float = 0.0
    rollback_min_score: float = 0.0
    shared_reward_pool_path: str = ""
    shared_pool_min_score: Optional[float] = None
    expose_env_name_to_llm: bool = False
    domain_knowledge_guard: bool = False
    forbidden_reward_terms: str = (
        "OFFICIAL_REWARD_MASKED,masked_reward,prev_shaping"
    )
    ppo_learning_rate: float = 2.5e-4
    ppo_n_steps: int = 1024
    ppo_batch_size: int = 64
    ppo_n_epochs: int = 4
    ppo_gamma: float = 0.999
    ppo_gae_lambda: float = 0.98
    ppo_clip_range: float = 0.2
    ppo_ent_coef: float = 0.01
    ppo_vf_coef: float = 0.5
    ppo_max_grad_norm: float = 0.5
    ppo_n_envs: int = 1
    ppo_verbose: int = 0
    ppo_device: str = "cpu"

    def ppo_kwargs(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.ppo_learning_rate,
            "n_steps": self.ppo_n_steps,
            "batch_size": self.ppo_batch_size,
            "n_epochs": self.ppo_n_epochs,
            "gamma": self.ppo_gamma,
            "gae_lambda": self.ppo_gae_lambda,
            "clip_range": self.ppo_clip_range,
            "ent_coef": self.ppo_ent_coef,
            "vf_coef": self.ppo_vf_coef,
            "max_grad_norm": self.ppo_max_grad_norm,
            "n_envs": self.ppo_n_envs,
            "verbose": self.ppo_verbose,
            "device": self.ppo_device,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        valid = {field.name for field in cls.__dataclass_fields__.values()}
        filtered = {key: value for key, value in data.items() if key in valid}
        return cls(**filtered)

    def with_overrides(self, **kwargs: Any) -> "ExperimentConfig":
        clean = {key: value for key, value in kwargs.items() if value is not None}
        return replace(self, **clean)


def load_config(path: Optional[str]) -> ExperimentConfig:
    if not path:
        return ExperimentConfig()

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        if config_path.suffix.lower() in {".json", ""}:
            data = json.load(handle)
        elif config_path.suffix.lower() in {".yaml", ".yml"}:
            try:
                import yaml  # type: ignore
            except ImportError as exc:
                raise RuntimeError("YAML config requires PyYAML. Use JSON or install pyyaml.") from exc
            data = yaml.safe_load(handle)
        else:
            raise ValueError(f"Unsupported config format: {config_path.suffix}")

    return ExperimentConfig.from_dict(data or {})
