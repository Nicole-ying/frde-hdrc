"""FDRE-HRDC prompt templates — Eureka-inspired clean structure + agent innovations."""
from __future__ import annotations
from textwrap import dedent
from typing import Optional


def build_initial_prompt(
    env_name: str, task_description: str,
    observation_description: str, action_description: str,
    reward_structure: str = "hrdc", expose_env_name: bool = False,
    eureka_context: str = "",
    allow_original_reward: bool = False,
) -> str:
    hrdc_rule = _hrdc_rule(reward_structure)
    reward_rule = _reward_rule(allow_original_reward)
    return dedent(f"""
    You are a reward engineer writing reward functions for RL tasks.
    Create a reward function to help the agent learn the task.
    Use only the provided observation and action variables.

    Task description:
    {task_description}

    Environment step source (shows how observations are constructed
    from the physics engine — the official reward is masked):
    {eureka_context}

    Write a Python function with this EXACT signature:

    def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
        ...
        return total_reward

    Coding rules:
    - Return a single float reward.
    - Use generic variable names (o=obs, n=next_obs).
    - Keep rewards numerically stable. Prefer smooth transformations.
    - {hrdc_rule}
    - {reward_rule}
    - Do not import modules. No try/except, classes, lambdas, file I/O, eval, exec.
    - Use only: obs, action, next_obs, info, training_progress.
    - Do not manually clamp the reward.
    - Output only the Python code.
    """).strip()


def build_refine_prompt(
    env_name: str, task_description: str,
    current_code: str, feedback: str,
    previous_best_code: Optional[str] = None,
    reward_structure: str = "hrdc", expose_env_name: bool = False,
    eureka_context: str = "", memory_context: str = "",
    allow_original_reward: bool = False,
    use_agent: bool = True,
) -> str:
    visible_env = env_name if expose_env_name else "BlackBoxControlEnv"
    hrdc_rule = _hrdc_rule(reward_structure)
    reward_rule = _reward_rule(allow_original_reward)
    best_block = f"\nBest reward code from previous iterations:\n{previous_best_code}\n" if previous_best_code else ""

    # ── 消融版：纯代码输出 ──
    if not use_agent:
        return dedent(f"""
        Improve this RL reward function. Output ONLY Python code.

        Rules: Output only code. No import. No try/except. No classes.
        Use only obs, action, next_obs, info, training_progress.
        {hrdc_rule} {reward_rule}

        Environment: {visible_env}
        Task: {task_description}

        Current code:
        {current_code}
        {best_block}
        Feedback:
        {feedback}

        {eureka_context}

        History:
        {memory_context}
        """).strip()

    # ── Agent 版：JSON 决策 + 骨架分析 ──
    return dedent(f"""
    You are a reward design agent. Your Agent Memory stores every reward function
    you've generated and its training outcome.

    ## Agent Memory
    {memory_context}

    ## Feedback
    {feedback}

    ## Current reward code
    {current_code}
    {best_block}

    ## Environment
    {visible_env}
    {task_description}
    {eureka_context}

    ## How to improve the reward

    SKELETON DIAGNOSIS — analyze before acting:

    1. INVENTORY: List every component and what physical signal it captures.
       Signal categories: transition progress, smoothness, action cost,
       stability (angle/angular velocity), velocity penalty, contact/landing, survival.

    2. MISSING SIGNALS: What is genuinely missing? Only flag a signal if the
       agent's poor performance directly points to it.
       - Episode < 200: likely missing stability or velocity signals, or undirected transition.
       - Episode max but score negative: directional signals point the wrong way.
       - If the skeleton already works (score > 100, episode > 500), do NOT add new components.

    3. REDUNDANCY & DIRECTION: Are two components measuring the same thing? Merge them.
       Is a component using undirected form (n-o)^2? Replace with abs(o)-abs(n).
       Is a component provably harmful (score dropped when it was added)? Delete it.

    4. STAGE WEIGHTS: Components should use training_progress for stage-based weights.
       Early training: low penalties (agent needs freedom to explore).
       Late training: high precision weights (contact, stability).

    5. DECIDE YOUR ACTION:
       - "add": genuinely missing signal(s) → add them.
       - "delete": harmful or redundant component(s) → remove them.
       - "tune": skeleton is adequate → adjust coefficients/directions/weights only.
       - "mix": multiple operations (add+delete, add+tune, delete+tune).
       - "rebuild": skeleton has been tried 2+ iterations and scores are flat/declining.
         This is a LAST RESORT. Only use when the skeleton is proven broken.

    ## Output format — first JSON, then code:

    ```json
    {{
      "action": "add" | "delete" | "tune" | "mix" | "rebuild",
      "reasoning": "Why this action? What did the skeleton diagnosis reveal?"
    }}
    ```

    ```python
    def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
        ...
        return total_reward
    ```

    Rules: Output only code. No import. No try/except. No classes.
    Use only: obs, action, next_obs, info, training_progress.
    {hrdc_rule} {reward_rule}
    Do not manually clamp the reward.
    """).strip()


def _hrdc_rule(reward_structure: str) -> str:
    if reward_structure == "static":
        return "Use fixed weights, do not vary with training_progress."
    if reward_structure == "flat":
        return "Use a single flat reward expression."
    return (
        "Use HRDC: each reward component has its own weight that evolves "
        "with training_progress. Early training (progress<0.3): prioritize exploration, "
        "keep penalties low. Middle (0.3-0.7): balance. Late (>=0.7): prioritize "
        "precision (contact, stability)."
    )


def _reward_rule(allow_original_reward: bool) -> str:
    if allow_original_reward:
        return "You may use original_reward as an anchor."
    return (
        "Do NOT reference original_reward. Derive the reward only from obs, action, "
        "next_obs, info, and training_progress."
    )
