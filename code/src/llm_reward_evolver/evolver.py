from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

from .agent_core import AgentAction, AgentDecision, AgentMemory, AgentState
from .config import ExperimentConfig
from .diagnostics import analyze_original_reward_anchor, detect_forbidden_domain_terms, uses_original_reward
from .eureka_context import load_eureka_context
from .feedback import build_feedback, build_scalar_feedback
from .llm import LLMClient, build_llm_client, extract_code
from .memory import RewardSearchMemory  # 保留旧 memory 兼容
from .prompts import build_initial_prompt, build_refine_prompt
from .reward import RewardProgram
from .trainer import inspect_env, make_env, train_agent


@dataclass
class IterationRecord:
    iteration: int
    score: float
    success_rate: float
    mean_episode_length: float
    converged: bool
    failure_mode: str
    reward_code_path: str
    interrupted: bool = False
    error_message: str = ""
    reward_error_count: int = 0
    reward_last_error: str = ""


class RewardEvolver:
    def __init__(self, config: ExperimentConfig, llm_client: Optional[LLMClient] = None) -> None:
        self.config = config
        self.llm_client = llm_client or build_llm_client(config.llm_provider, config.llm_model)
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self) -> List[IterationRecord]:
        observation_desc, action_desc = inspect_env(self.config.env_name, self.config.seed)
        eureka_context = load_eureka_context(
            self.config.env_name,
            self.config.task_description,
            self.config.task_description_path,
            self.config.step_source_path,
            self.config.mask_official_reward,
        )
        eureka_prompt_block = eureka_context.to_prompt_block()
        task_desc = eureka_context.task_description or task_desc
        memory = RewardSearchMemory(self.output_dir / "reward_search_memory.json")
        agent_memory = AgentMemory(self.output_dir / "agent_memory.json")  # 🆕
        records: List[IterationRecord] = []
        current_code: Optional[str] = None
        best_code: Optional[str] = None
        best_score = float("-inf")
        feedback = ""

        for iteration in range(self.config.max_iterations):
            restart = self._should_restart(records)
            if restart:
                current_code = None
                if best_score < self.config.rollback_min_score:
                    best_code = None
                feedback = (
                    "Structural restart: the skeleton has been tried for multiple iterations "
                    "and proven unfixable. Generate a completely new design from scratch."
                )

            if current_code is None:
                if iteration == 0 or not agent_memory.entries:
                    # 真正的 iter0：没有任何历史
                    prompt = build_initial_prompt(
                        self.config.env_name,
                        task_desc,
                        observation_desc,
                        action_desc,
                        self.config.reward_structure,
                        self.config.expose_env_name_to_llm,
                        eureka_prompt_block,
                        self.config.allow_original_reward_in_reward,
                    )
                else:
                    # 🆕 REBUILD：保留 memory + 骨架诊断，但不传旧代码
                    prompt = build_refine_prompt(
                        self.config.env_name,
                        task_desc,
                        "",  # 无 current_code
                        feedback,
                        "",  # 无 best_code
                        self.config.reward_structure,
                        self.config.expose_env_name_to_llm,
                        eureka_prompt_block,
                        agent_memory.render(),
                        self.config.allow_original_reward_in_reward,
                        use_agent=self.config.agent_mode,
                    )
            else:
                prompt = build_refine_prompt(
                    self.config.env_name,
                    task_desc,
                    current_code,
                    feedback,
                    best_code,
                    self.config.reward_structure,
                    self.config.expose_env_name_to_llm,
                    eureka_prompt_block,
                    agent_memory.render(),  # 🆕 Agent Memory
                    self.config.allow_original_reward_in_reward,
                    use_agent=self.config.agent_mode,  # 🆕 消融开关
                )

            fallback_code = None if restart and best_score < self.config.rollback_min_score else best_code
            # 保存完整 prompt 便于调试
            self._write_text(f"prompt_iter_{iteration}.txt", prompt)
            current_code, llm_response = self._generate_valid_reward(prompt, fallback_code)
            reward_path = self._write_text(f"reward_iter_{iteration}.py", current_code)
            # 🆕 提取 Agent 决策
            agent_decision = parse_agent_decision(llm_response, current_code)
            self._write_text(f"agent_decision_iter_{iteration}.json",
                json.dumps(agent_decision.to_dict(), ensure_ascii=False, indent=2))
            reward_program = RewardProgram(
                current_code,
                reward_clip=self.config.reward_clip,
                error_fallback=self.config.reward_error_fallback,
            )

            result = train_agent(
                self.config.env_name,
                reward_program,
                self.config.total_timesteps,
                self.config.eval_episodes,
                self.config.target_score,
                self.config.seed + iteration,
                self.config.training_algorithm,
                ppo_params=self.config.ppo_kwargs(),
            )

            stats = result.stats
            # 🆕 每轮保存PPO模型 + 最优模型
            model_path = self.output_dir / f"model_iter_{iteration}.zip"
            if result.model is not None:
                result.model.save(str(model_path))
            if stats.mean_eval_score > best_score:
                best_score = stats.mean_eval_score
                best_code = current_code
                self._write_text("reward_best.py", current_code)
                if result.model is not None:
                    result.model.save(str(self.output_dir / "model_best.zip"))
            if stats.mean_eval_score >= self._shared_pool_min_score():
                self._add_shared_pool_entry(current_code, stats.mean_eval_score, iteration)

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
            self._write_history(records)
            print(f"[seed={self.config.seed}] iter={iteration}/{self.config.max_iterations-1} "
                  f"score={stats.mean_eval_score:.1f} success={stats.success_rate:.2f} "
                  f"len={stats.mean_episode_length:.0f} err={stats.reward_error_count} "
                  f"best={best_score:.1f}", flush=True)
            memory.add(
                iteration=record.iteration,
                score=record.score,
                success_rate=record.success_rate,
                mean_episode_length=record.mean_episode_length,
                target_score=self.config.target_score,
                code_path=record.reward_code_path,
                failure_mode=record.failure_mode,
                code=current_code,  # 🆕 将本轮生成的完整代码写入memory
            )
            # 🆕 Agent Memory 记录
            agent_memory.record(
                iteration=iteration,
                action=agent_decision.action.value,
                target=agent_decision.target,
                reasoning=agent_decision.reasoning,
                score=stats.mean_eval_score,
                episode_length=stats.mean_episode_length,
                success_rate=stats.success_rate,
                code=current_code,
                code_path=str(reward_path),
                target_score=self.config.target_score,
            )

            feedback = self._build_feedback(stats, current_code)
            # 🆕 附加上一轮 LLM 的诊断
            if agent_memory.entries:
                last_entry = agent_memory.entries[-1]
                if last_entry.reasoning:
                    feedback += (
                        f"\nLast agent diagnosis: {last_entry.reasoning}"
                    )
            self._write_text(f"feedback_iter_{iteration}.txt", feedback)

            stop, _reason = self._should_stop(records)
            if stop:
                break

        return records

    def _build_feedback(self, stats, current_code: str = "") -> str:
        if self.config.feedback_mode == "scalar":
            return build_scalar_feedback(stats)
        feedback = build_feedback(stats)
        feedback += (
            "\nRules reminder:\n"
            "- No environment-specific semantics. Black-box MDP only.\n"
            "- Search for generic transition signals from the visible physics code.\n"
            "- If episode length is very short, the reward is unsafe.\n"
            "- If latest is worse than best but best is still poor, prefer structural restart.\n"
        )
        # 守卫检查
        if self.config.allow_original_reward_in_reward:
            anchor_report = analyze_original_reward_anchor(
                current_code, min_multiplier=self.config.min_original_reward_anchor)
            feedback += f"\n{anchor_report.to_feedback(self.config.min_original_reward_anchor)}"
        else:
            feedback += f"\n{self._reward_leakage_feedback(current_code, None)}"
        if self.config.domain_knowledge_guard:
            domain_hits = detect_forbidden_domain_terms(current_code, self.config.forbidden_reward_terms)
            feedback += f"\n{self._domain_feedback(domain_hits)}"
        return feedback

    def _generate_valid_reward(self, prompt: str, fallback_code: Optional[str]) -> tuple[str, str]:
        """返回 (code, llm_response)，方便后续提取 Agent 决策。"""
        last_error = ""
        last_response = ""
        for attempt in range(self.config.reward_repair_attempts + 1):
            response = self.llm_client.complete(prompt if attempt == 0 else self._repair_prompt(prompt, last_error))
            last_response = response
            code = extract_code(response)
            try:
                RewardProgram(
                    code,
                    reward_clip=self.config.reward_clip,
                    error_fallback=self.config.reward_error_fallback,
                )
                self._validate_reward_reference_policy(code)
                self._validate_domain_terms(code)
                self._smoke_test_reward_code(code)
                return code, response
            except Exception as exc:
                last_error = f"{type(exc).__name__}: {exc}"

        if fallback_code:
            try:
                self._validate_reward_reference_policy(fallback_code)
                self._validate_domain_terms(fallback_code)
                return fallback_code, last_response
            except ValueError:
                pass
        return default_reward_code(self.config.allow_original_reward_in_reward), last_response

    def _validate_reward_reference_policy(self, code: str) -> None:
        if not self.config.allow_original_reward_in_reward:
            if uses_original_reward(code):
                raise ValueError(
                    "Official reward leakage guard failed: generated reward code references "
                    "original_reward even though the official reward is masked."
                )
            return
        report = analyze_original_reward_anchor(
            code,
            min_multiplier=self.config.min_original_reward_anchor,
        )
        if report.suspicious:
            raise ValueError(report.to_feedback(self.config.min_original_reward_anchor))

    def _reward_leakage_feedback(self, code: str, anchor_report) -> str:
        if not self.config.allow_original_reward_in_reward:
            if uses_original_reward(code):
                return "- CRITICAL: official reward leakage guard detected original_reward usage. Remove it."
            return "- official reward leakage guard: original_reward is not used."
        return anchor_report.to_feedback(self.config.min_original_reward_anchor)

    def _validate_domain_terms(self, code: str) -> None:
        if not self.config.domain_knowledge_guard:
            return
        hits = detect_forbidden_domain_terms(code, self.config.forbidden_reward_terms)
        if hits:
            raise ValueError(
                "Domain-knowledge guard failed: reward code contains task-specific terms "
                f"{', '.join(hits)}. Use generic observation names and black-box transition signals."
            )

    def _domain_feedback(self, hits: List[str]) -> str:
        if not self.config.domain_knowledge_guard:
            return "- domain-knowledge guard: disabled."
        if not hits:
            return "- domain-knowledge guard: no forbidden task-specific terms detected."
        return (
            "- CRITICAL: domain-knowledge guard detected task-specific terms "
            f"{', '.join(hits)}. Rewrite using generic observation indices."
        )

    def _repair_prompt(self, original_prompt: str, error: str) -> str:
        return (
            f"{original_prompt}\n\n"
            "The previous reward code failed validation with this error:\n"
            f"{error}\n"
            "Return a corrected compute_reward function only."
        )

    def _smoke_test_reward_code(self, code: str) -> None:
        env = make_env(self.config.env_name, self.config.seed)
        try:
            obs, _info = env.reset(seed=self.config.seed)
            action = env.action_space.sample()
            next_obs, original_reward, _terminated, _truncated, info = env.step(action)
            program = RewardProgram(
                code,
                reward_clip=self.config.reward_clip,
                error_fallback=self.config.reward_error_fallback,
            )
            program(obs, action, next_obs, float(original_reward), info, 0.25)
            if program.error_count:
                raise ValueError(f"Reward runtime smoke test failed: {program.last_error}")
            self._generic_smoke_test(env, program, next_obs)
        finally:
            env.close()

    def _generic_smoke_test(self, env, program: RewardProgram, obs) -> None:
        values = []
        current_obs = obs
        for index, progress in enumerate((0.0, 0.5, 1.0)):
            action = env.action_space.sample()
            next_obs, original_reward, terminated, truncated, info = env.step(action)
            value = float(program(current_obs, action, next_obs, float(original_reward), info, progress))
            if not math.isfinite(value):
                raise ValueError("Reward runtime smoke test failed: non-finite reward")
            values.append(value)
            if terminated or truncated:
                current_obs, _info = env.reset(seed=self.config.seed + index + 1)
            else:
                current_obs = next_obs
        if program.error_count:
            raise ValueError(f"Reward runtime smoke test failed: {program.last_error}")

    def _should_stop(self, records: List[IterationRecord]) -> tuple[bool, Optional[str]]:
        latest = records[-1]
        if latest.score >= self.config.target_score:
            return True, "target score reached"
        if len(records) >= self.config.max_iterations:
            return True, "max iterations reached"
        if len(records) <= self.config.force_iterations_before_patience:
            return False, None
        if len(records) >= self.config.patience:
            recent = records[-self.config.patience :]
            improvements = [
                recent[i + 1].score - recent[i].score for i in range(len(recent) - 1)
            ]
            if all(value < self.config.min_improvement for value in improvements):
                return True, "patience reached"
        return False, None

    def _should_restart(self, records: List[IterationRecord]) -> bool:
        window = self.config.restart_after_bad_iterations
        if window <= 0 or len(records) < window:
            return False
        recent = records[-window:]
        return all(item.score < self.config.restart_score_threshold for item in recent)

    def _shared_pool_min_score(self) -> float:
        if self.config.shared_pool_min_score is not None:
            return float(self.config.shared_pool_min_score)
        return float(self.config.target_score)

    def _shared_pool_path(self) -> Optional[Path]:
        if not self.config.shared_reward_pool_path:
            return None
        return Path(self.config.shared_reward_pool_path)

    def _read_shared_pool(self) -> List[Dict[str, Any]]:
        path = self._shared_pool_path()
        if path is None or not path.exists():
            return []
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return []
        return data if isinstance(data, list) else []

    def _write_shared_pool(self, data: List[Dict[str, Any]]) -> None:
        path = self._shared_pool_path()
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")

    def _add_shared_pool_entry(self, code: str, score: float, iteration: int) -> None:
        path = self._shared_pool_path()
        if path is None:
            return
        data = self._read_shared_pool()
        if any(item.get("code") == code for item in data):
            return
        data.append(
            {
                "score": float(score),
                "seed": int(self.config.seed),
                "iteration": int(iteration),
                "code": code,
            }
        )
        data.sort(key=lambda item: float(item.get("score", float("-inf"))), reverse=True)
        self._write_shared_pool(data[:20])

    def _best_shared_pool_code(self) -> Optional[str]:
        for item in self._read_shared_pool():
            code = item.get("code")
            if isinstance(code, str):
                try:
                    self._validate_reward_reference_policy(code)
                    self._validate_domain_terms(code)
                    return code
                except ValueError:
                    continue
        return None

    def _write_text(self, name: str, text: str) -> Path:
        path = self.output_dir / name
        path.write_text(text, encoding="utf-8")
        return path

    def _write_history(self, records: List[IterationRecord]) -> None:
        path = self.output_dir / "history.json"
        data = [asdict(record) for record in records]
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def parse_agent_decision(response: str, code: str) -> AgentDecision:
    """从 LLM 响应中提取 Agent 的 JSON 决策。"""
    action_map = {
        "add": AgentAction.ADD, "delete": AgentAction.DELETE,
        "tune": AgentAction.TUNE, "mix": AgentAction.MIX,
        "rebuild": AgentAction.REBUILD,
    }
    # 尝试提取 JSON 块
    json_match = re.search(r'```json\s*(.*?)\s*```', response, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return AgentDecision(
                action=action_map.get(data.get("action", "tune"), AgentAction.TUNE),
                target=data.get("target", "skeleton"),
                reasoning=data.get("reasoning", ""),
                code=code,
                raw_response=response,
            )
        except (json.JSONDecodeError, KeyError):
            pass
    return AgentDecision(
        action=AgentAction.TUNE, target="unknown",
        reasoning="Could not parse JSON decision; inferred from code.",
        code=code, raw_response=response,
    )


def default_reward_code(allow_original_reward: bool = True) -> str:
    if allow_original_reward:
        return """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    return float(original_reward)
"""
    return """\
def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
    total = 0.0
    count = min(len(obs), len(next_obs))
    for i in range(count):
        total += abs(float(obs[i])) - abs(float(next_obs[i]))
    return float(total)
"""
