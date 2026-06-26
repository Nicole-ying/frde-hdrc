# FDRE-HRDC Delivery Package

This package contains the cleaned delivery materials for the FDRE-HRDC LunarLander-v3 experiments.

## Core Claim

FDRE-HRDC is presented as an autonomous reward-search agent, not a hand-written LunarLander reward design. The main pipeline uses generic observation/action interfaces, original environment reward, training logs, runtime errors, and evaluation scores to iteratively generate, test, diagnose, and refine reward functions. Environment-specific fixed rewards are retained only as legacy/canonical comparison and ablation references.

## Key Results

The final performance should be reported from the 1M-timestep setting. FDRE-HRDC solves LunarLander-v3 under 1M timesteps: all three seeds exceed the 200-point solved threshold, with seed scores `223.40`, `219.98`, and `234.74`. The mean score is `226.04 +/- 6.31`, and the mean solved episode rate is `0.82 +/- 0.06`.

The 100k-timestep setting is used as a fast reward-search and ablation study. Under this setting, FDRE-HRDC reaches `159.80 +/- 8.10`, clearly outperforming PPO baseline, LLM once, and both ablation variants. This supports the claim that diagnostic feedback and dynamic HRDC weighting improve sample efficiency before full convergence.

Reward runtime errors: `0`. Training interruptions caused by reward-function errors: `0`.

## Contents

- `figures/`: selected paper-ready single-panel figures.
- `data/`: JSON experiment summaries.
- `code/`: source-code snapshot and runnable scripts.
- `效果总结.md`: concise Chinese summary of current effects and recommended figures.
- `框架根因修复说明.md`: root-cause analysis and framework-level fixes for PPO variance, LLM prior leakage, weak anchor diagnosis, restart, and cross-seed sharing.
- `Eureka式流程修正说明.md`: corrected baseline/FDRE workflow using task file, masked step source, prompt, leakage guard, and memory.
- `最终复验说明.md`: final smoke-verification record and long-run validation command.
- `FDRE-HRDC公式化交付说明.md`: Typora-compatible Chinese formula document.
- `FDRE-HRDC_formula_delivery.md`: same content with an ASCII filename.
- `交付包目录.md`: Chinese delivery index.
- `delivery_index.md`: English delivery index.

The source code is provided directly in `code/`; no separate code archive is included.
