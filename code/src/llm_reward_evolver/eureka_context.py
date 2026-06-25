from __future__ import annotations

import inspect
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional


REWARD_MASK = "<OFFICIAL_REWARD_MASKED>"


@dataclass(frozen=True)
class EurekaContext:
    task_description: str
    step_source: str

    def to_prompt_block(self) -> str:
        return (
            "Task description file:\n"
            f"{self.task_description.strip()}\n\n"
            "Environment step source with official reward masked:\n"
            f"{self.step_source.strip()}\n"
        )


def load_eureka_context(
    env_name: str,
    task_description: str,
    task_description_path: str = "",
    step_source_path: str = "",
    mask_official_reward: bool = True,
) -> EurekaContext:
    task_text = _read_text_if_present(task_description_path) or task_description
    step_text = _read_text_if_present(step_source_path) or _default_step_source(env_name)
    if mask_official_reward:
        step_text = mask_reward_logic(step_text)
    return EurekaContext(task_description=task_text, step_source=step_text)


def mask_reward_logic(source: str) -> str:
    """Mask obvious official reward construction while preserving step mechanics."""

    lines = source.splitlines()
    masked: list[str] = []
    inserted = False
    in_reward_block = False

    for line in lines:
        stripped = line.strip()
        indent = len(line) - len(line.lstrip())

        if in_reward_block:
            if stripped.startswith("return "):
                masked.append(_mask_return_reward(line))
                in_reward_block = False
            continue

        if re.match(r"reward\s*=", stripped) or "reward =" in stripped:
            if not inserted:
                masked.append(" " * indent + "# Official reward computation is intentionally masked.")
                masked.append(" " * indent + f"masked_reward = {REWARD_MASK}")
                inserted = True
            in_reward_block = True
            continue

        if stripped.startswith("return "):
            masked.append(_mask_return_reward(line))
        else:
            masked.append(line)

    return "\n".join(masked)


def _mask_return_reward(line: str) -> str:
    return re.sub(r"\breward\b", "masked_reward", line)


def _read_text_if_present(path: str) -> str:
    if not path:
        return ""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Context file does not exist: {path}")
    return file_path.read_text(encoding="utf-8")


def _default_step_source(env_name: str) -> str:
    name = env_name.lower()
    if name.startswith("lunarlander"):
        try:
            import gymnasium.envs.box2d.lunar_lander as lunar_lander

            return inspect.getsource(lunar_lander.LunarLander.step)
        except Exception:
            return _fallback_step_stub(env_name)
    return _fallback_step_stub(env_name)


def _fallback_step_stub(env_name: str) -> str:
    return f"""\
def step(self, action):
    \"\"\"Fallback source stub for {env_name}; official reward is unavailable here.\"\"\"
    next_obs, official_reward, terminated, truncated, info = self.env.step(action)
    reward = {REWARD_MASK}
    return next_obs, masked_reward, terminated, truncated, info
"""
