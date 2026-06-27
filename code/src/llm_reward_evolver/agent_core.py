"""Agent core: state, actions, and decision loop.

The agent operates in a perceive→plan→act→observe→remember loop.
It has four explicit actions and maintains structured memory of all past decisions.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional


# ── Agent Actions ──────────────────────────────────────────────

class AgentAction(str, Enum):
    ADD     = "add"       # 添加缺失的信号类别
    DELETE  = "delete"    # 删除有害/冗余的组件
    TUNE    = "tune"      # 微调系数（不增删组件类型）
    MIX     = "mix"       # 同时做了 ADD + DELETE + TUNE 中的多种
    REBUILD = "rebuild"   # 骨架经2+轮迭代仍无改善，彻底抛弃重建


# ── Agent Decision ─────────────────────────────────────────────

@dataclass
class AgentDecision:
    """LLM 做出的一个决策。"""
    action: AgentAction
    target: str                     # 操作目标："skeleton" | "height_penalty" | "contact_weight"
    reasoning: str                  # LLM 的诊断理由
    code: str                       # 生成的新奖励代码
    raw_response: str = ""          # LLM 原始返回（调试用）

    def to_dict(self) -> dict:
        return {
            "action": self.action.value,
            "target": self.target,
            "reasoning": self.reasoning,
        }


# ── Agent State ────────────────────────────────────────────────

@dataclass
class AgentState:
    """Agent 在每个时刻的完整状态。"""
    iteration: int = 0
    current_code: Optional[str] = None
    best_code: Optional[str] = None
    best_score: float = float("-inf")
    last_score: float = 0.0
    last_episode_length: float = 0.0
    last_error_count: int = 0
    target_score: float = 200.0
    is_terminal: bool = False
    stop_reason: str = ""

    def perceive(self, memory: "AgentMemory") -> str:
        """感知：整合当前状态 + 历史记忆 → 文本描述。"""
        parts = [
            f"Iteration: {self.iteration}",
            f"Last score: {self.last_score:.1f} | Episode length: {self.last_episode_length:.0f}",
            f"Best score: {self.best_score:.1f} | Target: {self.target_score:.0f}",
            f"Errors: {self.last_error_count}",
            "",
            memory.render(),
        ]
        return "\n".join(parts)


# ── Agent Memory ───────────────────────────────────────────────

@dataclass
class MemoryEntry:
    """记忆中的一条记录——一次完整的 agent 决策循环。"""
    iteration: int
    action: str          # "rebuild" | "delete" | "add" | "tune" | "initial"
    target: str          # 操作目标
    reasoning: str       # 决策理由
    score: float         # 执行后的得分
    episode_length: float
    success_rate: float
    code: str            # 完整奖励代码
    verdict: str         # "success" | "failure"
    code_path: str = ""


class AgentMemory:
    """Agent 的记忆系统——存全量代码 + 支持演化查询。"""

    def __init__(self, path: Optional[Path] = None) -> None:
        self.path = path
        self.entries: List[MemoryEntry] = []
        if path and path.exists():
            self._load()

    def record(
        self,
        iteration: int,
        action: str,
        target: str,
        reasoning: str,
        score: float,
        episode_length: float,
        success_rate: float,
        code: str,
        code_path: str = "",
        target_score: float = 200.0,
    ) -> None:
        verdict = "success" if score >= target_score else "failure"
        self.entries.append(MemoryEntry(
            iteration=iteration, action=action, target=target,
            reasoning=reasoning, score=score, episode_length=episode_length,
            success_rate=success_rate, code=code, verdict=verdict,
            code_path=code_path,
        ))
        if self.path:
            self._write()

    def render(self) -> str:
        """渲染记忆为 Agent 可读的文本——全量代码 + 演化追踪。"""
        if not self.entries:
            return "Agent Memory: no experience yet."

        lines = [
            "=== Agent Memory ===",
            "",
            "## Evolution Summary",
            "| iter | action | target | score | len | verdict | diagnosis (LLM-generated) |",
            "|------|--------|--------|-------|-----|---------|---------------------------|",
        ]
        for e in self.entries:
            lesson = e.reasoning[:80] + ("..." if len(e.reasoning) > 80 else "")
            lines.append(
                f"| {e.iteration} | {e.action} | {e.target} | "
                f"{e.score:.1f} | {e.episode_length:.0f} | {e.verdict} | {lesson} |"
            )
        lines.append("")

        # 全量代码：每轮完整展示
        for e in self.entries:
            lines.append(
                f"## Iteration {e.iteration} — {e.action.upper()} on {e.target} "
                f"(score={e.score:.1f}, verdict={e.verdict})"
            )
            if e.reasoning:
                lines.append(f"Decision reasoning: {e.reasoning}")
            lines.append("```python")
            lines.append(e.code.strip())
            lines.append("```")
            lines.append("")

        return "\n".join(lines)

    def query_best(self) -> Optional[MemoryEntry]:
        """查询最佳记忆。"""
        if not self.entries:
            return None
        return max(self.entries, key=lambda e: e.score)

    def query_recent(self, n: int = 3) -> List[MemoryEntry]:
        """查询最近 n 条记忆。"""
        return self.entries[-n:]

    def query_by_action(self, action: str) -> List[MemoryEntry]:
        """查询特定 action 类型的记忆。"""
        return [e for e in self.entries if e.action == action]

    def trend(self, last_n: int = 5) -> str:
        """分析最近 n 轮的趋势。"""
        recent = self.entries[-last_n:]
        if len(recent) < 2:
            return "Not enough data for trend analysis."
        scores = [e.score for e in recent]
        if all(scores[i] < scores[i+1] for i in range(len(scores)-1)):
            return "Trend: IMPROVING (monotonic increase)"
        if all(scores[i] > scores[i+1] for i in range(len(scores)-1)):
            return "Trend: DECLINING (monotonic decrease)"
        if max(scores) - min(scores) < 20:
            return f"Trend: STAGNANT (all within {max(scores)-min(scores):.0f} points)"
        return "Trend: OSCILLATING (no clear direction)"

    def _load(self) -> None:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            for item in data:
                item.setdefault("action", "unknown")
                item.setdefault("target", "")
                item.setdefault("reasoning", "")
                self.entries.append(MemoryEntry(**item))
        except Exception:
            pass

    def _write(self) -> None:
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(
                json.dumps([e.__dict__ for e in self.entries], ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
