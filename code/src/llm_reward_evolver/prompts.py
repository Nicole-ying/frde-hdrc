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
    use_agent: bool = True,
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
            "HOW TO IMPROVE THE REWARD FUNCTION — follow this exact process:\n"
            "\n"
            "1. ANALYZE THE CURRENT SKELETON:\n"
            "   a) List every component and what physical signal it guides. "
            "Signal types: transition progress, smoothness, action cost, stability "
            "(angle, angular velocity), velocity penalty, contact/landing, survival.\n"
            "   b) Identify what is MISSING. If episode < 200: likely missing stability "
            "or velocity signals. If episode is max but score negative: directional "
            "signals point the wrong way.\n"
            "   c) Check for REDUNDANCY. If two components measure the same thing "
            "(e.g. both abs(o)-abs(n)), merge them. If a component uses undirected "
            "form like (n-o)^2, it must be replaced with directional form abs(o)-abs(n).\n"
            "\n"
            "2. DECIDE WHAT TO DO:\n"
            "   - If a signal category is MISSING: use ADD to fill it.\n"
            "   - If a component is REDUNDANT or HARMFUL: use DELETE to remove it.\n"
            "   - If the skeleton is COMPLETE but poorly calibrated: use TUNE to adjust "
            "coefficients, fix signal directions, or rebalance stage weights.\n"
            "   - If the skeleton has been given at least 2 iterations of genuine "
            "improvement attempts (the parent skeleton has 2+ child iterations in "
            "Memory) AND scores are flat or declining despite those attempts: REBUILD.\n"
            "   - If scores show a clear upward trend across iterations, do NOT "
            "rebuild — keep improving with ADD/DELETE/TUNE.\n"
            "\n"
            "3. OUTPUT YOUR DECISION as JSON with your component analysis, "
            "then generate the improved code. Your changes must be justified "
            "by the analysis above.\n"
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
    if not use_agent:
        # ── 消融版：无 Agent 结构，直接生成代码 ──
        return dedent(
            f"""
            Improve the following RL reward function based on training evidence.
            Study the memory and feedback below. Execute the skeleton diagnosis steps.
            Output ONLY the improved Python code — no JSON, no explanation.

            Rules:
            - Output only Python code.
            - Do not import modules. Do not use try/except, classes, lambdas.
            - Use only {allowed_inputs}.
            - {structure_rule}
            - {reward_rule}
            - Do not manually clamp the reward.
            - For discrete actions, treat action as a scalar integer.{analysis_task}

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
          "inventory": ["comp_name: what it measures", ...],
          "missing": ["signal_category_not_present", ...],
          "harmful": ["comp_name: why harmful", ...],
          "action": "add" | "delete" | "tune" | "rebuild",
          "target": "specific_component_name",
          "reasoning": "Based on inventory/missing/harmful analysis"
        }}
        ```
        ACTION RULES for the \"action\" field:
        - If your new code keeps 2+ components from the current code → use \"add\", \"delete\", or \"tune\".
          You CANNOT use \"rebuild\" if the skeleton structure is partially preserved.
        - Only use \"rebuild\" if you COMPLETELY replaced everything — zero components reused.
        FILLING \"inventory\", \"missing\", \"harmful\" IS MANDATORY.

        ```python
        def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
            ...
        ```

        ACTION MEANINGS — choose the one that BEST describes what you actually did:
        - "add": you added one or more missing signal categories to the EXISTING skeleton.
          The original skeleton's core structure is preserved.
        - "delete": you removed one or more harmful/redundant components from the EXISTING skeleton.
        - "tune": you adjusted coefficients, fixed signal directions, or rebalanced weights
          within the EXISTING skeleton, without adding or removing component types.
        - "rebuild": you COMPLETELY replaced the skeleton with a fundamentally different design.
          None of the original component structure is reused. Use this ONLY when the skeleton
          has been given 2+ iterations of genuine improvement attempts and scores are
          flat or declining — proving the skeleton itself is the problem.
          IMPORTANT: if you keep 2+ components from the original skeleton, it is NOT a rebuild —
          it is ADD, DELETE, or TUNE.

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
