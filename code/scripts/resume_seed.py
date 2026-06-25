"""Resume FDRE-HRDC experiment for seed_44 from iter3."""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
sys.path.insert(0, str(SRC))

from llm_reward_evolver.config import ExperimentConfig, load_config
from llm_reward_evolver.evolver import RewardEvolver, IterationRecord
from llm_reward_evolver.llm import build_llm_client
from llm_reward_evolver.memory import RewardSearchMemory
from llm_reward_evolver.prompts import build_refine_prompt
from llm_reward_evolver.eureka_context import load_eureka_context
from llm_reward_evolver.trainer import inspect_env, make_env, train_agent
from llm_reward_evolver.feedback import build_feedback
from llm_reward_evolver.reward import RewardProgram


def resume_seed(config_path: str, seed: int, start_iter: int, llm_provider: str, llm_model: str):
    base_config = load_config(config_path)
    seed_dir = Path(base_config.output_dir) / f"seed_{seed}"

    config = base_config.with_overrides(
        seed=seed,
        output_dir=str(seed_dir),
        shared_reward_pool_path="",
    )

    llm = build_llm_client(llm_provider, llm_model)
    evolver = RewardEvolver(config, llm)

    # 加载已有状态
    history_path = seed_dir / "history.json"
    memory_path = seed_dir / "reward_search_memory.json"

    old_records = [IterationRecord(**r) for r in json.loads(history_path.read_text(encoding="utf-8"))]
    memory = RewardSearchMemory(memory_path)

    # 恢复 best
    best_record = max(old_records, key=lambda r: r.score)
    best_score = best_record.score
    best_code = (seed_dir / f"reward_iter_{best_record.iteration}.py").read_text(encoding="utf-8")
    current_code = (seed_dir / f"reward_iter_{old_records[-1].iteration}.py").read_text(encoding="utf-8")

    print(f"Resuming seed={seed} from iter={start_iter}")
    print(f"  History: {len(old_records)} iters, best=iter{best_record.iteration} score={best_score:.1f}")
    print(f"  Last: iter{old_records[-1].iteration} score={old_records[-1].score:.1f}")

    # 手动跑后续迭代
    observation_desc, action_desc = inspect_env(config.env_name, config.seed)
    eureka_context = load_eureka_context(
        config.env_name, config.task_description,
        config.task_description_path, config.step_source_path, config.mask_official_reward,
    )
    eureka_prompt_block = eureka_context.to_prompt_block()
    records = list(old_records)

    for iteration in range(start_iter, config.max_iterations):
        # 构建 feedback（基于上一轮）
        last = records[-1]
        stats_obj = type('obj', (object,), {
            'mean_eval_score': last.score,
            'success_rate': last.success_rate,
            'mean_episode_length': last.mean_episode_length,
            'trend': 'needs_improvement',
            'converged': last.converged,
            'failure_mode': last.failure_mode,
            'interrupted': last.interrupted,
            'error_message': last.error_message,
            'reward_error_count': last.reward_error_count,
            'reward_last_error': last.reward_last_error,
        })()

        feedback = evolver._build_feedback(stats_obj, current_code)

        # 🆕 结构性重启：连续N轮低于阈值 → 推倒重建
        restart = evolver._should_restart(records)
        if restart:
            current_code = None
            if best_score < config.rollback_min_score:
                best_code = None
            feedback = (
                "Structural restart triggered: recent iterations stayed below the failure "
                "threshold. Generate a FRESH reward structure from scratch. Do NOT reuse any "
                "previous code — the old skeleton is proven insufficient. Start over with "
                "at least 5-6 signal categories and a proper 3-stage HRDC weight structure.\n"
            )

        # 保守回退逻辑
        if not restart and best_code and best_score > last.score and best_score >= config.rollback_min_score:
            feedback += "\nConservative rollback: start from previous_best_code, not latest code.\n"

        if current_code is None:
            from llm_reward_evolver.prompts import build_initial_prompt
            prompt = build_initial_prompt(
                config.env_name, config.task_description,
                observation_desc, action_desc,
                config.reward_structure, config.expose_env_name_to_llm,
                eureka_prompt_block, config.allow_original_reward_in_reward,
            )
        else:
            prompt = build_refine_prompt(
                config.env_name, config.task_description,
                current_code, feedback, best_code,
                config.reward_structure, config.expose_env_name_to_llm,
                eureka_prompt_block, memory.to_prompt_block(),
                config.allow_original_reward_in_reward,
            )

        fallback = None if (evolver._should_restart(records) and best_score < config.rollback_min_score) else best_code
        current_code = evolver._generate_valid_reward(prompt, fallback)
        reward_path = seed_dir / f"reward_iter_{iteration}.py"
        reward_path.write_text(current_code, encoding="utf-8")

        reward_program = RewardProgram(current_code, reward_clip=config.reward_clip, error_fallback=config.reward_error_fallback)

        result = train_agent(
            config.env_name, reward_program, config.total_timesteps,
            config.eval_episodes, config.target_score,
            config.seed + iteration, config.training_algorithm,
            ppo_params=config.ppo_kwargs(),
        )

        stats = result.stats
        if stats.mean_eval_score > best_score:
            best_score = stats.mean_eval_score
            best_code = current_code
            (seed_dir / "reward_best.py").write_text(current_code, encoding="utf-8")

        if result.model is not None:
            result.model.save(str(seed_dir / f"model_iter_{iteration}.zip"))
            if stats.mean_eval_score >= best_score:
                result.model.save(str(seed_dir / "model_best.zip"))

        record = IterationRecord(
            iteration=iteration,
            score=stats.mean_eval_score,
            success_rate=stats.success_rate,
            mean_episode_length=stats.mean_episode_length,
            converged=stats.converged,
            failure_mode=stats.failure_mode,
            reward_code_path=str(reward_path),
            interrupted=stats.interrupted,
            error_message=stats.error_message,
            reward_error_count=stats.reward_error_count,
            reward_last_error=stats.reward_last_error,
        )
        records.append(record)

        # 写 history
        from dataclasses import asdict
        history_path.write_text(json.dumps([asdict(r) for r in records], ensure_ascii=False, indent=2), encoding="utf-8")

        memory.add(
            iteration=iteration, score=stats.mean_eval_score,
            success_rate=stats.success_rate, mean_episode_length=stats.mean_episode_length,
            target_score=config.target_score, code_path=str(reward_path),
            failure_mode=stats.failure_mode, code=current_code,
        )

        feedback_text = evolver._build_feedback(stats, current_code)
        (seed_dir / f"feedback_iter_{iteration}.txt").write_text(feedback_text, encoding="utf-8")

        print(f"[seed={seed}] iter={iteration}/{config.max_iterations-1} "
              f"score={stats.mean_eval_score:.1f} success={stats.success_rate:.2f} "
              f"len={stats.mean_episode_length:.0f} best={best_score:.1f}", flush=True)

        stop, reason = evolver._should_stop(records)
        if stop:
            print(f"Stopped: {reason}")
            break

    print(f"\nSeed {seed} done. Best score={best_score:.1f}")


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--seed", type=int, default=44)
    p.add_argument("--start-iter", type=int, default=3)
    p.add_argument("--llm-provider", default="deepseek")
    p.add_argument("--llm-model", default="deepseek-chat")
    args = p.parse_args()
    resume_seed(args.config, args.seed, args.start_iter, args.llm_provider, args.llm_model)
