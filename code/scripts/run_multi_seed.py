"""Run FDRE-HRDC independently across multiple seeds (no shared pool)."""
from __future__ import annotations

import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from llm_reward_evolver.config import load_config
from llm_reward_evolver.evolver import RewardEvolver
from llm_reward_evolver.llm import build_llm_client


def main() -> None:
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--llm-provider", default="deepseek")
    parser.add_argument("--llm-model", default="deepseek-chat")
    parser.add_argument("--seeds", default="42,43,44", help="Comma-separated seeds")
    args = parser.parse_args()

    base_config = load_config(args.config)
    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    llm = build_llm_client(args.llm_provider, args.llm_model)

    all_results: dict[str, list] = {}

    for seed in seeds:
        seed_dir = Path(base_config.output_dir) / f"seed_{seed}"
        seed_config = base_config.with_overrides(
            seed=seed,
            output_dir=str(seed_dir),
            shared_reward_pool_path="",  # 强制禁用共享池
        )
        print(f"\n{'='*60}")
        print(f"Starting seed={seed}, output={seed_dir}")
        print(f"{'='*60}")
        records = RewardEvolver(seed_config, llm).run()
        all_results[str(seed)] = [
            {
                "iteration": r.iteration,
                "score": r.score,
                "success_rate": r.success_rate,
                "mean_episode_length": r.mean_episode_length,
                "converged": r.converged,
                "failure_mode": r.failure_mode,
                "interrupted": r.interrupted,
                "reward_error_count": r.reward_error_count,
            }
            for r in records
        ]
        best = max(records, key=lambda r: r.score)
        print(f"Seed {seed} done: best_iter={best.iteration}, best_score={best.score:.3f}")

    # 写汇总
    summary_path = Path(base_config.output_dir) / "all_seeds_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(all_results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSummary written to {summary_path}")


if __name__ == "__main__":
    main()
