from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
PACK = ROOT / "deliverables" / "完整交付包"
FIG = PACK / "figures"
SUMMARY_1M = PACK / "data" / "fdre_hrdc_stable_1m_summary.json"
SUMMARY_100K = PACK / "data" / "lunarlander_100k_suite_summary.json"
HISTORY_ROOT = ROOT / "outputs" / "lunarlander_paper"

FDRE = "#2374AB"
FDRE_DARK = "#174A7C"
GREEN = "#2E8B57"
GRAY = "#AEB7C2"
GRAY_DARK = "#687382"
RED = "#B94A48"
AMBER = "#D49A3A"
ROSE = "#C97979"
DARK = "#20242A"
GRID = "#E8EDF3"
SPINE = "#C8D0DA"


def load(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def setup() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    for path in FIG.glob("*.png"):
        path.unlink()
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 180,
            "savefig.dpi": 420,
            "font.family": "DejaVu Sans",
            "axes.labelsize": 8.0,
            "xtick.labelsize": 7.2,
            "ytick.labelsize": 7.2,
            "legend.fontsize": 7.0,
            "axes.edgecolor": SPINE,
            "axes.linewidth": 0.75,
            "grid.color": GRID,
            "grid.linewidth": 0.55,
            "grid.alpha": 0.95,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def clean(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(SPINE)
    ax.spines["bottom"].set_color(SPINE)
    ax.tick_params(colors=DARK, width=0.75, length=3)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    fig.tight_layout()
    fig.savefig(FIG / name, bbox_inches="tight", pad_inches=0.035)
    plt.close(fig)


def by_method(summary):
    return {row["method"]: row for row in summary}


def method_label(method: str) -> str:
    return {
        "baseline_original_reward": "PPO baseline",
        "llm_once": "LLM once",
        "ablation_no_diagnostic_feedback": "w/o diagnostic",
        "ablation_no_dynamic_weights": "w/o dynamic",
        "fdre": "FDRE-HRDC",
    }[method]


def parse_series(value):
    return json.loads(value) if isinstance(value, str) else value


def history_mean(method: str, metric: str):
    paths = sorted((HISTORY_ROOT / method).glob("seed_*/history.json"))
    if not paths:
        return np.array([]), np.array([])
    rows = []
    for path in paths:
        records = load(path)
        rows.append([float(item[metric]) for item in records])
    width = max(len(row) for row in rows)
    grid = np.full((len(rows), width), np.nan)
    for index, row in enumerate(rows):
        grid[index, : len(row)] = row
    return np.arange(width), np.nanmean(grid, axis=0)


def draw_threshold(ax, y=200.0) -> None:
    ax.axhline(y, color=RED, lw=0.9, ls=(0, (3, 2)))
    ax.text(0.99, y + 1.4, "solved threshold", transform=ax.get_yaxis_transform(), ha="right", va="bottom", fontsize=6.8, color=RED)


def plot_1m_seed_scores(stable):
    scores = np.asarray(stable["seed_scores"], dtype=float)
    seeds = [str(row["seed"]) for row in stable["records"]]
    fig, ax = plt.subplots(figsize=(3.35, 2.35))
    x = np.arange(len(scores))
    ax.vlines(x, 200, scores, color=GREEN, lw=5.5, alpha=0.28)
    ax.scatter(x, scores, s=54, color=GREEN, edgecolor="white", linewidth=0.75, zorder=3)
    draw_threshold(ax)
    ax.axhline(stable["mean_score"], color=FDRE_DARK, lw=1.0)
    ax.text(2.36, stable["mean_score"] + 1.4, f"mean {stable['mean_score']:.1f}", fontsize=6.8, color=FDRE_DARK, ha="right")
    for xi, score in zip(x, scores):
        ax.text(xi, score + 2.2, f"{score:.1f}", ha="center", fontsize=7.0, color=DARK)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Original return")
    ax.set_ylim(195, 241)
    clean(ax)
    save(fig, "01_1M_seed_scores.png")


def plot_1m_success(stable):
    rates = np.asarray(stable["seed_success_rates"], dtype=float)
    seeds = [str(row["seed"]) for row in stable["records"]]
    fig, ax = plt.subplots(figsize=(3.2, 2.25))
    x = np.arange(len(rates))
    bars = ax.bar(x, rates, width=0.48, color=GREEN, alpha=0.86)
    ax.axhline(stable["success_rate"], color=FDRE_DARK, lw=1.0)
    ax.text(2.35, stable["success_rate"] + 0.025, f"mean {stable['success_rate']:.2f}", fontsize=6.8, color=FDRE_DARK, ha="right")
    ax.bar_label(bars, [f"{v:.2f}" for v in rates], fontsize=7.0, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels(seeds)
    ax.set_xlabel("Seed")
    ax.set_ylabel("Solved episode rate")
    ax.set_ylim(0, 1.02)
    clean(ax)
    save(fig, "03_1M_success_rate.png")


def plot_100k_score(summary):
    data = by_method(summary)
    methods = [
        "baseline_original_reward",
        "llm_once",
        "ablation_no_diagnostic_feedback",
        "ablation_no_dynamic_weights",
        "fdre",
    ]
    values = np.asarray([data[m]["mean_score"] for m in methods], dtype=float)
    errors = np.asarray([data[m]["score_std"] for m in methods], dtype=float)
    colors = [GRAY, GRAY_DARK, ROSE, AMBER, FDRE]
    fig, ax = plt.subplots(figsize=(4.25, 2.65))
    x = np.arange(len(methods))
    bars = ax.bar(x, values, yerr=errors, width=0.56, color=colors, alpha=0.92, capsize=2.4, error_kw={"elinewidth": 0.8, "capthick": 0.8})
    ax.axhline(0, color=DARK, lw=0.75)
    ax.bar_label(bars, [f"{v:.1f}" for v in values], fontsize=6.7, padding=2)
    ax.annotate("+164.4 vs PPO", xy=(4, values[-1]), xytext=(2.95, values[-1] + 50), fontsize=7.0, color=FDRE_DARK, arrowprops=dict(arrowstyle="-|>", lw=0.75, color=FDRE_DARK))
    ax.set_xticks(x)
    ax.set_xticklabels([method_label(m).replace(" ", "\n") for m in methods])
    ax.set_ylabel("Original return")
    ax.set_ylim(-145, 245)
    clean(ax)
    save(fig, "04_100k_score_comparison.png")


def plot_100k_success(summary):
    data = by_method(summary)
    methods = [
        "baseline_original_reward",
        "llm_once",
        "ablation_no_diagnostic_feedback",
        "ablation_no_dynamic_weights",
        "fdre",
    ]
    values = np.asarray([data[m]["success_rate"] for m in methods], dtype=float)
    colors = [GRAY, GRAY_DARK, ROSE, AMBER, FDRE]
    fig, ax = plt.subplots(figsize=(3.8, 2.5))
    y = np.arange(len(methods))
    bars = ax.barh(y, values, height=0.46, color=colors, alpha=0.92)
    ax.bar_label(bars, [f"{v:.2f}" for v in values], fontsize=6.9, padding=3)
    ax.set_yticks(y)
    ax.set_yticklabels([method_label(m) for m in methods])
    ax.set_xlabel("Solved episode rate")
    ax.set_xlim(0, 0.92)
    clean(ax)
    save(fig, "05_100k_success_rate.png")


def plot_ablation_gain(summary):
    data = by_method(summary)
    fdre_score = float(data["fdre"]["mean_score"])
    refs = [
        ("PPO baseline", float(data["baseline_original_reward"]["mean_score"]), GRAY_DARK),
        ("LLM once", float(data["llm_once"]["mean_score"]), GRAY_DARK),
        ("w/o diagnostic", float(data["ablation_no_diagnostic_feedback"]["mean_score"]), ROSE),
        ("w/o dynamic", float(data["ablation_no_dynamic_weights"]["mean_score"]), AMBER),
    ]
    gains = np.asarray([fdre_score - score for _, score, _ in refs], dtype=float)
    fig, ax = plt.subplots(figsize=(3.75, 2.45))
    y = np.arange(len(refs))
    bars = ax.barh(y, gains, height=0.48, color=[color for _, _, color in refs], alpha=0.92)
    ax.bar_label(bars, [f"+{v:.1f}" for v in gains], fontsize=6.9, padding=3)
    ax.set_yticks(y)
    ax.set_yticklabels([name for name, _, _ in refs])
    ax.set_xlabel("Return improvement of FDRE-HRDC")
    ax.set_xlim(0, gains.max() * 1.15)
    clean(ax)
    save(fig, "06_ablation_score_gain.png")


def plot_seed_distribution(summary, stable):
    data = by_method(summary)
    methods = [
        "baseline_original_reward",
        "llm_once",
        "ablation_no_diagnostic_feedback",
        "ablation_no_dynamic_weights",
        "fdre",
    ]
    values = [np.asarray(parse_series(data[m]["seed_scores"]), dtype=float) for m in methods]
    values.append(np.asarray(stable["seed_scores"], dtype=float))
    labels = ["PPO", "LLM", "no diag", "static", "FDRE\n100k", "FDRE\n1M"]
    colors = [GRAY, GRAY_DARK, ROSE, AMBER, FDRE, GREEN]
    fig, ax = plt.subplots(figsize=(4.35, 2.75))
    for index, (vals, color) in enumerate(zip(values, colors), start=1):
        jitter = np.linspace(-0.07, 0.07, len(vals))
        ax.scatter(np.full(len(vals), index) + jitter, vals, s=24, color=color, edgecolor="white", linewidth=0.55, zorder=3)
        ax.plot([index - 0.18, index + 0.18], [vals.mean(), vals.mean()], color=color, lw=2.0)
        ax.vlines(index, vals.min(), vals.max(), color=color, lw=1.0, alpha=0.55)
    draw_threshold(ax)
    ax.set_xticks(np.arange(1, len(labels) + 1))
    ax.set_xticklabels(labels)
    ax.set_ylabel("Original return")
    ax.set_ylim(-150, 250)
    clean(ax)
    save(fig, "07_seed_level_distribution.png")


def plot_reward_iteration_score(summary):
    data = by_method(summary)
    specs = [
        ("llm_once", "LLM once", GRAY_DARK),
        ("ablation_no_diagnostic_feedback", "w/o diagnostic", ROSE),
        ("ablation_no_dynamic_weights", "w/o dynamic", AMBER),
        ("fdre", "FDRE-HRDC", FDRE),
    ]
    fig, ax = plt.subplots(figsize=(4.0, 2.55))
    for method, label, color in specs:
        xs, ys = history_mean(method, "score")
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, marker="o", ms=3.4, lw=1.55 if method == "fdre" else 1.0, color=color, alpha=0.95 if method == "fdre" else 0.68, label=label)
    fdre_final = float(data["fdre"]["mean_score"])
    ax.scatter([3], [fdre_final], s=42, color=GREEN, edgecolor="white", linewidth=0.65, zorder=4)
    ax.text(3, fdre_final + 17, f"selected {fdre_final:.1f}", ha="center", fontsize=6.8, color=GREEN)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0", "1", "2", "selected"])
    ax.set_xlabel("Reward-search iteration")
    ax.set_ylabel("Mean return")
    ax.legend(frameon=False, loc="lower right")
    clean(ax)
    save(fig, "08_reward_iteration_score.png")


def plot_reward_iteration_success(summary):
    data = by_method(summary)
    specs = [
        ("llm_once", "LLM once", GRAY_DARK),
        ("ablation_no_diagnostic_feedback", "w/o diagnostic", ROSE),
        ("ablation_no_dynamic_weights", "w/o dynamic", AMBER),
        ("fdre", "FDRE-HRDC", FDRE),
    ]
    fig, ax = plt.subplots(figsize=(4.0, 2.55))
    for method, label, color in specs:
        xs, ys = history_mean(method, "success_rate")
        if len(xs) == 0:
            continue
        ax.plot(xs, ys, marker="o", ms=3.4, lw=1.55 if method == "fdre" else 1.0, color=color, alpha=0.95 if method == "fdre" else 0.68, label=label)
    fdre_final = float(data["fdre"]["success_rate"])
    ax.scatter([3], [fdre_final], s=42, color=GREEN, edgecolor="white", linewidth=0.65, zorder=4)
    ax.text(3, fdre_final + 0.045, f"selected {fdre_final:.2f}", ha="center", fontsize=6.8, color=GREEN)
    ax.set_xticks([0, 1, 2, 3])
    ax.set_xticklabels(["0", "1", "2", "selected"])
    ax.set_xlabel("Reward-search iteration")
    ax.set_ylabel("Solved episode rate")
    ax.set_ylim(-0.02, 0.94)
    ax.legend(frameon=False, loc="upper left")
    clean(ax)
    save(fig, "09_reward_iteration_success.png")


def plot_evidence_chain(summary, stable):
    data = by_method(summary)
    labels = ["PPO\n100k", "LLM once\n100k", "FDRE\n100k", "FDRE\n1M"]
    scores = np.asarray(
        [
            data["baseline_original_reward"]["mean_score"],
            data["llm_once"]["mean_score"],
            data["fdre"]["mean_score"],
            stable["mean_score"],
        ],
        dtype=float,
    )
    colors = [GRAY, GRAY_DARK, FDRE, GREEN]
    fig, ax = plt.subplots(figsize=(3.9, 2.65))
    x = np.arange(len(scores))
    ax.plot(x, scores, color=SPINE, lw=1.1, zorder=1)
    ax.scatter(x, scores, s=52, color=colors, edgecolor="white", linewidth=0.65, zorder=2)
    draw_threshold(ax)
    for xi, score, color in zip(x, scores, colors):
        offset = 13 if score < 200 else 8
        ax.text(xi, score + offset, f"{score:.1f}", ha="center", fontsize=7.0, color=color)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Original return")
    ax.set_ylim(-95, 252)
    clean(ax)
    save(fig, "11_evidence_chain.png")


def main() -> None:
    setup()
    stable = load(SUMMARY_1M)
    summary = load(SUMMARY_100K)
    plot_1m_seed_scores(stable)
    plot_1m_success(stable)
    plot_100k_score(summary)
    plot_100k_success(summary)
    plot_ablation_gain(summary)
    plot_seed_distribution(summary, stable)
    plot_reward_iteration_score(summary)
    plot_reward_iteration_success(summary)
    plot_evidence_chain(summary, stable)
    print(FIG)


if __name__ == "__main__":
    main()
