from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import List


@dataclass
class MemoryEntry:
    iteration: int
    score: float
    success_rate: float
    mean_episode_length: float
    verdict: str
    lesson: str
    code_path: str
    code: str = ""  # 🆕 存储完整奖励函数代码


class RewardSearchMemory:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.entries: List[MemoryEntry] = self._load()

    def add(
        self,
        iteration: int,
        score: float,
        success_rate: float,
        mean_episode_length: float,
        target_score: float,
        code_path: str,
        failure_mode: str,
        code: str = "",  # 🆕 接收代码
    ) -> None:
        verdict = "success" if score >= target_score else "failure"
        lesson = _lesson_from_metrics(score, success_rate, mean_episode_length, target_score, failure_mode)
        self.entries.append(
            MemoryEntry(
                iteration=iteration,
                score=float(score),
                success_rate=float(success_rate),
                mean_episode_length=float(mean_episode_length),
                verdict=verdict,
                lesson=lesson,
                code_path=code_path,
                code=code,
            )
        )
        self._write()

    def to_prompt_block(self, max_entries: int = 10) -> str:
        """🆕 输出每轮完整代码 + 分数 + 教训，供 LLM 做组件级对比分析"""
        if not self.entries:
            return "Reward-search memory: no previous experience yet."

        selected = self.entries[-max_entries:]
        lines = [
            "Reward-search memory (historical reward functions and their outcomes):",
            "",
        ]

        # 先输出汇总表
        lines.append("## Historical Summary Table")
        lines.append("| iter | score | success | length | verdict | key lesson |")
        lines.append("|------|-------|---------|--------|---------|-------------|")
        for item in selected:
            lines.append(
                f"| {item.iteration} | {item.score:.3f} | {item.success_rate:.3f} | "
                f"{item.mean_episode_length:.0f} | {item.verdict} | {item.lesson} |"
            )
        lines.append("")

        # 再输出每轮的完整代码
        for item in selected:
            lines.append(f"## Iteration {item.iteration} (score={item.score:.3f}, verdict={item.verdict})")
            lines.append("```python")
            lines.append(item.code if item.code else f"# code saved to {item.code_path}")
            lines.append("```")
            lines.append("")

        # 🆕 分析指令
        if len(selected) >= 2:
            lines.append("## Component Analysis Task")
            lines.append(
                "Before generating the next reward function, you MUST analyze the historical data above:\n"
                "1. Compare reward functions across iterations: what components (terms, penalties, bonuses, "
                "stage weights) were added, removed, or modified between each iteration?\n"
                "2. For each change, note whether the score improved or degraded. Identify which "
                "components correlated with better scores and which correlated with worse scores.\n"
                "3. Based on this analysis, decide: should you (a) keep and tune the best components, "
                "(b) remove harmful components, or (c) try a structurally different approach?\n"
                "4. Generate the next reward function based on this evidence, not on guesswork.\n"
            )

        return "\n".join(lines)

    def _load(self) -> List[MemoryEntry]:
        if not self.path.exists():
            return []
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except Exception:
            return []
        # 兼容旧格式（没有 code 字段的旧记录）
        entries = []
        for item in data:
            if isinstance(item, dict):
                item.setdefault("code", "")
                entries.append(MemoryEntry(**item))
        return entries

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(
            json.dumps([asdict(item) for item in self.entries], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )


def _lesson_from_metrics(
    score: float,
    success_rate: float,
    mean_episode_length: float,
    target_score: float,
    failure_mode: str,
) -> str:
    if score >= target_score:
        return "Keep this reward structure as a successful candidate; future edits should be conservative."
    if mean_episode_length < 100:
        return "Failure experience: episodes terminate very early; avoid reward structures that create unsafe behavior."
    if success_rate == 0.0:
        return "Failure experience: no solved episodes; a structural reward rewrite is preferred over small coefficient edits."
    return f"Failure experience: {failure_mode}; adjust generic transition features or stage weights."
