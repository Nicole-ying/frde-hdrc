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
            "   - If a signal category is MISSING AND a component is HARMFUL: use MIX.\n"
            "     Example: add stability+velocity+survival AND delete undirected sq_change = MIX.\n"
            "   - If ONLY missing signals → use ADD. If ONLY harmful/redundant → use DELETE.\n"
            "   - If the skeleton is COMPLETE but poorly calibrated: use TUNE.\n"
            "   - REBUILD is the LAST RESORT. Only use it when this skeleton has been "
            "tried for 2+ iterations AND scores are flat/declining. In ALL other cases, "
            "use MIX, ADD, DELETE, or TUNE. When in doubt, choose MIX — it preserves "
            "the good parts while fixing the bad.\n"
            "   - If scores show a clear upward trend, do NOT rebuild.\n"
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
          "action": "add" | "delete" | "tune" | "mix" | "rebuild",
          "reasoning": "Walk through your analysis: what components exist, what (if anything) is genuinely missing or harmful, and why you chose this action."
        }}
        ```
        ACTIONS:
        - "add"    = you added genuinely missing signal categories
        - "delete" = you removed proven harmful or redundant components
        - "tune"   = you adjusted coefficients/signals only, no component changes
        - "mix"    = you did two or more of add/delete/tune
        - "rebuild"= LAST RESORT. Skeleton tried 2+ iterations, still flat/declining.
                     Discard memory, start fresh as iter0.

        ```python
        def compute_reward(obs, action, next_obs, original_reward, info, training_progress=0.0):
            ...
        ```

        ACTION MEANINGS:
        - "add": you added missing signal categories. Skeleton preserved.
        - "delete": you removed harmful/redundant components. Skeleton preserved.
        - "tune": you adjusted coefficients/signs/weights only. No component type changes.
        - "mix": you did BOTH adding missing signals AND removing harmful ones (or tuning).
          Example: the skeleton has 4 components, you add 3 new signal types (stability,
          velocity, survival) AND delete 1 harmful component (undirected sq_change) AND
          tune coefficients. This is MIX — you are improving the existing skeleton, not
          discarding it. MIX is the most common action, especially in early iterations.
          When in doubt between MIX and REBUILD: if you are building ON TOP of the
          existing skeleton concept, it's MIX. Only if you are starting a fundamentally
          different design philosophy is it REBUILD.
        - "rebuild": LAST RESORT. The skeleton has been tried for 2+ iterations and
          proven unfixable. Discard it entirely, clear memory context, and generate
          a fresh design from scratch as if at iter0. Use sparingly.

        Keep the same signature.

        Rules:
        - Output only Python code.
        - Do not import modules.
        - Do not use try/except, classes, lambdas, file I/O, eval, exec, or external libraries.
        - Use only {allowed_inputs}.
        - {structure_rule}
        - Fix the failure mode described by the feedback.
        - {reward_rule}
        - Follow the HOW TO IMPROVE analysis process above to decide your action.
        - If adding missing signal categories, use ADD. If removing harmful ones, use DELETE.
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
