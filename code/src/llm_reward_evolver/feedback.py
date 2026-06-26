from __future__ import annotations

from dataclasses import dataclass
from statistics import mean
from typing import Dict, Iterable, List


@dataclass
class EpisodeSummary:
    reward: float
    length: int
    success: bool = False
    final_x: float = 0.0
    start_x: float = 0.0


@dataclass
class TrainingStats:
    mean_eval_score: float
    success_rate: float
    mean_episode_length: float
    trend: str
    converged: bool
    failure_mode: str
    interrupted: bool = False
    error_message: str = ""
    reward_error_count: int = 0
    reward_last_error: str = ""

    def to_dict(self) -> Dict[str, object]:
        return {
            "mean_eval_score": self.mean_eval_score,
            "success_rate": self.success_rate,
            "mean_episode_length": self.mean_episode_length,
            "trend": self.trend,
            "converged": self.converged,
            "failure_mode": self.failure_mode,
            "interrupted": self.interrupted,
            "error_message": self.error_message,
            "reward_error_count": self.reward_error_count,
            "reward_last_error": self.reward_last_error,
        }


def summarize_episodes(episodes: Iterable[EpisodeSummary], target_score: float) -> TrainingStats:
    data = list(episodes)
    if not data:
        return TrainingStats(0.0, 0.0, 0.0, "no_data", False, "")
    rewards = [item.reward for item in data]
    lengths = [item.length for item in data]
    success_rate = sum(item.success for item in data) / len(data)
    mean_score = mean(rewards)
    mean_length = mean(lengths)
    trend = "good" if mean_score >= target_score else "needs_improvement"
    converged = mean_score >= target_score or success_rate >= 0.8
    return TrainingStats(mean_score, success_rate, mean_length, trend, converged, "")


def build_feedback(stats: TrainingStats) -> str:
    """只返回数据，不做诊断。诊断交给 LLM。"""
    per_step = stats.mean_eval_score / max(1, stats.mean_episode_length)
    return (
        f"Score: {stats.mean_eval_score:.1f} "
        f"| Success rate: {stats.success_rate:.2f} "
        f"| Episode length: {stats.mean_episode_length:.0f} "
        f"| Per-step: {per_step:.3f} "
        f"| Errors: {stats.reward_error_count}"
    )


def build_scalar_feedback(stats: TrainingStats) -> str:
    return f"Mean evaluation score: {stats.mean_eval_score:.3f}."
