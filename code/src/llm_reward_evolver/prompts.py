from __future__ import annotations

from textwrap import dedent
from typing import Optional


def build_initial_prompt(
    env_name: str,
    task_description: str,
    observation_description: str,
    action_description: str,
    reward_structure: str = "hrdc",
    expose_env_name: bool = False,
    eureka_context: str = "",
    allow_original_reward: bool = False,
) -> str:
    structure_rule = _structure_rule(reward_structure)
    visible_env = env_name if expose_env_name else "BlackBoxControlEnv"
    reward_rule = _official_reward_rule(allow_original_reward)
    allowed_inputs = (
        "obs, action, next_obs, original_reward, info, and training_progress"
        if allow_original_reward
        else "obs, action, next_obs, info, and training_progress"
    )
    signal_rule = (
        "Search for reusable signals from obs, next_obs, action, original_reward, info, and training_progress only."
        if allow_original_reward
        else "Search for reusable signals from obs, next_obs, action, info, and training_progress only."
    )
    return dedent(
        f"""
        You are an autonomous reinforcement-learning reward-search agent.
        Generate one Python function named compute_reward with this exact signature:

        def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
            ...

        Requirements:
        - Return a single float reward.
        - Do not import modules.
        - Do not use try/except, classes, lambdas, file I/O, eval, exec, or external libraries.
        - Use only {allowed_inputs}.
        - {structure_rule}
        - {reward_rule}
        - Keep the reward numerically stable.
        - Use only information present in the task description and the masked step source.
        - Do not reconstruct, approximate, or copy the masked official reward formula.
        - {signal_rule}
        - A good black-box pattern is to create generic arrays o and n, then use dimension-agnostic
          transition features such as sum(abs(o[i]) - abs(n[i])), sum((n[i] - o[i]) ** 2),
          small action cost, and stage weights. Keep variable names generic.
        - Do not manually clamp the reward inside compute_reward; the training framework performs final clipping.
        - Output only Python code.

        Environment: {visible_env}
        Observation interface: {observation_description}
        Action interface: {action_description}
        Task: {task_description}

        Eureka-style context:
        {eureka_context}
        """
    ).strip()


def build_refine_prompt(
    env_name: str,
    task_description: str,
    current_code: str,
    feedback: str,
    previous_best_code: Optional[str] = None,
    reward_structure: str = "hrdc",
    expose_env_name: bool = False,
    eureka_context: str = "",
    memory_context: str = "",
    allow_original_reward: bool = False,
) -> str:
    best_block = f"\nPrevious best reward code:\n{previous_best_code}\n" if previous_best_code else ""
    structure_rule = _structure_rule(reward_structure)
    visible_env = env_name if expose_env_name else "BlackBoxControlEnv"
    reward_rule = _official_reward_rule(allow_original_reward)
    allowed_inputs = (
        "obs, action, next_obs, original_reward, info, and training_progress"
        if allow_original_reward
        else "obs, action, next_obs, info, and training_progress"
    )
    def _analysis_task(memory_ctx: str, feedback_text: str) -> str:
        """Build skeleton-quality diagnosis and component analysis instructions."""
        skeleton_diagnosis = (
            "\n\n"
            "SKELETON QUALITY DIAGNOSIS (do this BEFORE writing code):\n"
            "\n"
            "Step 0 — Is this skeleton worth fixing? Look at the TREND, not the level:\n"
            "  - Good rewards often start from very negative scores and climb. "
            "If scores are IMPROVING (even from -400 to -200 to -100), the skeleton "
            "is working — keep tuning. The trend matters more than the current value.\n"
            "  - If scores are FLAT or DECLINING across 3+ iterations despite "
            "coefficient changes: the skeleton is stuck. REBUILD with a different "
            "design. Don't waste more iterations on it.\n"
            "  - If the agent NEVER survives past 200 steps in any iteration: the "
            "reward is too punishing. REBUILD with lower early-stage penalties.\n"
            "  - If there was at least ONE iteration with episode > 500 or score "
            "positive: the skeleton CAN work. It just needs calibration.\n"
            "\n"
            "Step 1 — Inventory the skeleton. List EVERY component in the current reward function "
            "and what physical signal it captures:\n"
            "  - Transition signal (e.g. abs_change, movement): does the agent move toward desirable states?\n"
            "  - Smoothness penalty: does it prevent jerky/jumpy behavior?\n"
            "  - Action cost: does it penalize unnecessary engine use?\n"
            "  - Stability signal (angle, angular velocity): does it prevent tumbling/spinning?\n"
            "  - Contact/landing signal (leg contact, height): does it reward successful touchdown?\n"
            "  - Velocity penalty: does it discourage crashing at high speed?\n"
            "  - Survival bonus: does it reward staying alive longer?\n"
            "\n"
            "Step 2 — Identify what is MISSING. Look at the feedback metrics:\n"
            "  - Compute per-step reward: mean_score / mean_episode_length. "
            "If this is very negative (< -0.5 per step), your reward signal is "
            "fundamentally punishing the agent. Check all signs and directions.\n"
            "  - If episode_length is short (< 200): the agent dies quickly. "
            "You are likely MISSING a stability signal (angle/angular-velocity penalty) "
            "or your transition signal is undirected (rewarding chaotic movement).\n"
            "  - If episode_length is max (1000) but score is very negative (< -50): "
            "the agent survives but does the WRONG thing. Your directional signals "
            "likely point the wrong way (e.g. rewarding movement away from center).\n"
            "  - If score oscillates wildly between iterations: "
            "your components may CONFLICT or your reward scale per step is too large.\n"
            "\n"
            "Step 3 — Check signal DIRECTION, SCALE, and REDUNDANCY (MANDATORY — do NOT skip):\n"
            "  - DIRECTION: Each component must point toward desirable behavior. "
            "abs(o[i])-abs(n[i]) is directional (toward zero). (n[i]-o[i])**2 is UNDIRECTED "
            "(rewards ANY change). If a component uses an undirected form, replace it with "
            "a directional one.\n"
            "  - SCALE: Estimate per-step contribution. A constant survival_bonus of 0.5 "
            "gives +500 over 1000 steps, drowning all other signals. Keep per-component "
            "per-step contribution in ±0.05 to ±2.0. Total per-step reward should be ±5 max.\n"
            "  - REDUNDANCY (CRITICAL): Compare EVERY pair of components. If two components "
            "measure the SAME physical quantity, you MUST merge them into one. Examples:\n"
            "    * abs_change = sum(abs(o)-abs(n)) AND progress_signal = sum(abs(o)-abs(n)) "
            "→ IDENTICAL, merge.\n"
            "    * vel_change = sum(|n-o|) AND sq_change = sum((n-o)^2) → both measure "
            "state-change magnitude (L1 vs L2), redundant. Keep the one that correlates "
            "better with scores.\n"
            "    * Many reward functions succeed with 7-9 components. Count alone is NOT "
            "a problem — only TRUE redundancy is. Compare every pair carefully.\n"
            "  - HARMFUL SIGNS: Does a component appear with a NEGATIVE sign in late training? "
            "That may be punishing desirable behavior.\n"
            "  - DELETE AUTHORITY: You CAN and SHOULD delete components. If a component was "
            "added in the previous iteration and the score dropped significantly, DELETE it "
            "— do NOT just reduce its weight to zero or near-zero. Actually remove it from "
            "the code. If a component provides no unique signal, DELETE it. If a component "
            "consistently correlates with worse scores, DELETE it. A cleaner reward function "
            "with fewer well-chosen components is better than a bloated one.\n"
            "  - When deciding between lowering a weight and deleting a component: if the "
            "component measures something that SHOULD matter (e.g. contact for landing), "
            "tune the weight. If the component is REDUNDANT with another, HARMFUL based on "
            "score evidence, or conceptually WRONG, DELETE it.\n"
            "\n"
            "SIMPLE-FIRST PRINCIPLE:\n"
            "  - Start every new skeleton with the SIMPLEST design that covers the "
            "necessary signal categories. 4-6 clean components with clear stage "
            "weights is better than 7-9 components with subtle interactions.\n"
            "  - Only add a new component when there is clear evidence the skeleton "
            "needs it (scores flat despite coefficient tuning across 2+ iterations).\n"
            "  - Simple skeletons are easy to debug and improve. Complex skeletons "
            "have too many interacting parameters — hard to tell what works.\n"
            "  - CRITICAL: If the current iteration scored WELL (close to or above "
            "the target), make ONLY one small coefficient adjustment. Do NOT add "
            "new components, do NOT add new penalties to early training. A score "
            "near the target means the skeleton is RIGHT — don't break it.\n"
            "  - Early training stage (stage 1 / early_weight) MUST keep penalties "
            "LOW. The agent needs freedom to explore before precision matters. "
            "Never add new penalty terms to the early stage of a working reward.\n"
            "\n"
            "SKELETON QUALITY — YOU decide whether to REBUILD:\n"
            "  - You have access to EVERY iteration's complete code and score in "
            "Agent Memory. Look at the score trajectory across all iterations.\n"
            "  - Ask yourself: is there a REAL trend of improvement (not just "
            "noise)? Real improvement means scores consistently moving in the "
            "right direction across multiple iterations, regardless of magnitude. "
            "If scores went -200 → -150 → -100 → -50, that IS real improvement — "
            "keep tuning. If scores went -100 → -110 → -105 → -115, that's just "
            "noise around -110 — the skeleton is stuck, REBUILD.\n"
            "  - OBVIOUSLY BROKEN: undirected signals like (n-o)^2, ALL scores "
            "deeply negative with episode < 200 across all iterations, penalties "
            "heavy in early training. REBUILD immediately — don't waste iterations.\n"
            "  - IMPROVING: scores show a genuine upward trend. Keep searching. "
            "Don't rebuild just because scores are still negative — progress is "
            "progress. Trust the trend.\n"
            "  - STUCK: scores oscillate around the same level for multiple "
            "iterations with no clear trend. REBUILD with a different structure.\n"
            "\n"
            "Step 5 — Output the improved reward function. Your changes must be justified "
            "by the skeleton diagnosis above. If the skeleton is insufficient, ADD components. "
            "If it is adequate, TUNE components."
        )

        if not memory_ctx or "no previous experience" in memory_ctx:
            return (
                skeleton_diagnosis
                + "\n\n"
                "Since this is the first refinement with no historical comparison yet, "
                "focus on Step 1-2: inventory the current skeleton and add any missing "
                "signal categories based on the feedback failure mode."
            )

        return (
            skeleton_diagnosis
            + "\n\n"
            "CROSS-ITERATION EVIDENCE:\n"
            "Study the Agent Memory section below. For each past iteration, note its "
            "components and its score. If the same skeleton was tried with different "
            "coefficients and scores stayed bad → the skeleton is the problem, not the coefficients. "
            "If adding a component made scores worse → that component is harmful, remove it. "
            "If simplifying the skeleton made scores better → the original skeleton was over-engineered."
        )

    analysis_task = _analysis_task(memory_context, feedback)
    return dedent(
        f"""
        You are an Autonomous Reward Design Agent. You operate in a perceive→plan→act loop.
        Your Agent Memory stores every reward function you've ever generated and its score.

        ## Your Task

        Study the Agent Memory below. Execute the SKELETON QUALITY DIAGNOSIS.
        Then output your decision in this EXACT format — first a JSON decision block,
        then the Python code:

        ```json
        {{
          "action": "rebuild" | "delete" | "add" | "tune",
          "target": "skeleton" or the specific component name,
          "reasoning": "Why you chose this action, based on evidence from Memory"
        }}
        ```

        ```python
        def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
            ...
        ```

        ACTION MEANINGS:
        - "rebuild": skeleton is broken → generate a fresh simple design from scratch
        - "delete": a specific component is harmful/redundant → remove it
        - "add": a signal category is missing → add the needed component
        - "tune": skeleton is working → adjust coefficients/stage-weights

        Keep the same signature.

        Rules:
        - Output only Python code.
        - Do not import modules.
        - Do not use try/except, classes, lambdas, file I/O, eval, exec, or external libraries.
        - Use only {allowed_inputs}.
        - {structure_rule}
        - Fix the failure mode described by the feedback.
        - {reward_rule}
        - Follow the SKELETON QUALITY DIAGNOSIS above to decide whether to rebuild or tune.
        - If the diagnosis says REBUILD, add missing signal categories freely — do not be conservative.
        - If the diagnosis says TUNE, make targeted coefficient/sign/stage-weight adjustments.
        - Do not manually clamp the reward inside compute_reward; the training framework performs final clipping.
        - Avoid undefined variables. Before returning, check that every variable has been assigned.
        - For discrete actions, treat action as a scalar integer, not an array.
        - Do not introduce rules not justified by task description, masked step source, feedback, or memory.
        - Prefer generic reward-search patterns: transition progress, smoothness, action cost, stage weights,
          and numerical safety.
        - If reward leakage guard failed, remove any use of masked official reward variables or original_reward.{analysis_task}

        Environment: {visible_env}
        Task: {task_description}

        Current reward code:
        {current_code}
        {best_block}
        FDRE feedback / diagnostic report:
        {feedback}

        Eureka-style context:
        {eureka_context}

        Agent memory:
        {memory_context}
        """
    ).strip()


def _structure_rule(reward_structure: str) -> str:
    if reward_structure == "static":
        return (
            "Use decomposed reward components with fixed weights only; do not change weights "
            "based on training_progress or state. This is an ablation without dynamic weighting."
        )
    if reward_structure == "flat":
        return "Use a single flat reward expression without HRDC decomposition. This is an ablation."
    return (
        "Use an HRDC-style structure: propose generic reward components from observed "
        "transition signals, then combine them with stage weights based on training_progress. "
        "Each component should have its own weight that evolves across training stages."
    )


def _official_reward_rule(allow_original_reward: bool) -> str:
    if allow_original_reward:
        return (
            "Preserve the original task objective: use original_reward directly as the dominant, "
            "unscaled anchor term; do not multiply original_reward by a coefficient smaller than 0.5."
        )
    return (
        "The official reward is masked and must not be used. Do not reference original_reward in "
        "the generated reward code; derive the training reward only from obs, action, next_obs, "
        "info, and training_progress."
    )
