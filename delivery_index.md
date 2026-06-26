# Delivery Index

## Delivery Positioning

This package is organized around autonomous, domain-knowledge-free reward search. The latest framework revision addresses three observed failure sources: PPO training variance, LLM prior leakage from environment-specific task descriptions, and failed seeds being trapped by weak `original_reward` anchors.

## Documents

- `README.md`
- `效果总结.md`
- `框架根因修复说明.md`
- `Eureka式流程修正说明.md`
- `最终复验说明.md`
- `FDRE-HRDC公式化交付说明.md`
- `FDRE-HRDC_formula_delivery.md`
- `交付包目录.md`
- `delivery_index.md`

## Selected Figures

- `figures/01_1M_seed_scores.png`
- `figures/03_1M_success_rate.png`
- `figures/04_100k_score_comparison.png`
- `figures/05_100k_success_rate.png`
- `figures/06_ablation_score_gain.png`
- `figures/07_seed_level_distribution.png`
- `figures/08_reward_iteration_score.png`
- `figures/09_reward_iteration_success.png`
- `figures/11_evidence_chain.png`

## Data

- `data/fdre_hrdc_stable_1m_summary.json`
- `data/lunarlander_100k_suite_summary.json`
- `data/lunarlander_smoke_verify_history.json`
- `data/lunarlander_smoke_verify_reward_iter_0.py`
- `data/lunarlander_smoke_verify_feedback_iter_0.txt`

## Code

- `code/src/llm_reward_evolver/`
- `code/src/llm_reward_evolver/diagnostics.py`
- `code/scripts/`
- `code/configs/lunarlander_robust_1m.json`
- `code/configs/lunarlander_smoke_verify.json`
- `code/tests/test_diagnostics.py`
- `code/requirements.txt`
- `code/pyproject.toml`
- `code/code_delivery.md`
