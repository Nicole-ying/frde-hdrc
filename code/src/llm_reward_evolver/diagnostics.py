from __future__ import annotations

import ast
from dataclasses import dataclass
from typing import List, Optional


@dataclass(frozen=True)
class RewardAnchorReport:
    min_multiplier: Optional[float]
    suspicious: bool
    messages: List[str]

    def to_feedback(self, min_required: float) -> str:
        if not self.messages:
            return (
                f"- original_reward anchor check: no explicit down-scaling below {min_required:.3g} "
                "was detected."
            )
        return "\n".join(f"- {message}" for message in self.messages)


def analyze_original_reward_anchor(code: str, min_multiplier: float = 0.5) -> RewardAnchorReport:
    """Detect obvious cases where original_reward is weakened too much.

    The check is intentionally conservative and task-agnostic. It does not try to
    understand the reward semantics; it only prevents the search from disabling
    the environment objective via patterns such as original_reward * 0.001.
    """

    try:
        tree = ast.parse(code)
    except SyntaxError as exc:
        return RewardAnchorReport(None, True, [f"anchor check skipped because code is invalid: {exc}"])

    constants = _constant_assignments(tree)
    multipliers: List[float] = []
    direct_uses = 0
    for node in ast.walk(tree):
        if _is_original_reward(node):
            direct_uses += 1
        value = _original_reward_multiplier(node, constants)
        if value is not None:
            multipliers.append(value)

    messages: List[str] = []
    min_seen = min(multipliers) if multipliers else None
    if direct_uses == 0:
        messages.append(
            "CRITICAL: original_reward is not used. Restore the environment objective with "
            "reward = original_reward + shaping_terms."
        )
        return RewardAnchorReport(min_seen, True, messages)

    if min_seen is not None and min_seen < min_multiplier:
        messages.append(
            f"CRITICAL: original_reward appears to be scaled by {min_seen:.3g}, below the "
            f"required {min_multiplier:.3g}. Use original_reward directly or at least "
            f"{min_multiplier:.3g} * original_reward."
        )
        return RewardAnchorReport(min_seen, True, messages)

    return RewardAnchorReport(min_seen, False, messages)


def detect_forbidden_domain_terms(code: str, forbidden_terms: str) -> List[str]:
    """Detect task-semantics terms that would leak human/domain prior into code."""

    terms = [term.strip().lower() for term in forbidden_terms.split(",") if term.strip()]
    if not terms:
        return []
    lower_code = code.lower()
    hits = sorted({term for term in terms if term in lower_code})
    return hits


def uses_original_reward(code: str) -> bool:
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False
    return any(_is_original_reward(node) for node in ast.walk(tree))


def _is_original_reward(node: ast.AST) -> bool:
    return isinstance(node, ast.Name) and node.id == "original_reward"


def _original_reward_multiplier(node: ast.AST, constants: dict[str, float]) -> Optional[float]:
    if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Mult):
        return None
    left_is_anchor = _is_original_reward(node.left)
    right_is_anchor = _is_original_reward(node.right)
    if left_is_anchor:
        return _literal_float(node.right, constants)
    if right_is_anchor:
        return _literal_float(node.left, constants)
    return None


def _constant_assignments(tree: ast.AST) -> dict[str, float]:
    constants: dict[str, float] = {}
    for node in ast.walk(tree):
        if not isinstance(node, ast.Assign) or len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name):
            value = _literal_float(node.value, constants)
            if value is not None:
                constants[target.id] = value
    return constants


def _literal_float(node: ast.AST, constants: dict[str, float]) -> Optional[float]:
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return abs(float(node.value))
    if isinstance(node, ast.Name):
        return constants.get(node.id)
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.USub, ast.UAdd)):
        value = _literal_float(node.operand, constants)
        return value if value is None else abs(value)
    return None
