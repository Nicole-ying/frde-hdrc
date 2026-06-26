"""Generate TWO complete figure sets from FDRE-HRDC experiments.
Set A: v2 3-seed (skeleton diagnosis, all solved)
Set B: 10-seed final (full validation, 5/10 solved)
"""
import json, glob
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
OUT.mkdir(exist_ok=True)
for old in OUT.glob("*.png"):
    old.unlink()

SOLVED = 200
BASELINE = 260.4
FS = 12

plt.rcParams.update({
    "figure.dpi": 200, "savefig.dpi": 300,
    "font.size": FS, "axes.labelsize": FS,
    "axes.titlesize": 14, "xtick.labelsize": 10, "ytick.labelsize": 10,
    "legend.fontsize": 9, "figure.facecolor": "white",
})

BLUE = '#2E86AB'
RED = '#D64045'
GREEN = '#2E8B57'
PURPLE = '#A23B72'
GRAY = '#808080'
ORANGE = '#F18F01'

def load(exp_dir, seed):
    f = glob.glob(str(ROOT / "outputs" / exp_dir / f"seed_{seed}" / "history.json"))
    return json.load(open(f[0])) if f else []

def load_best(exp_dir, seed):
    d = load(exp_dir, seed)
    return max(r['score'] for r in d) if d else None

# ═══════════════════════════════════════════════════
#  SET A: v2 3-seed experiment
# ═══════════════════════════════════════════════════
EXP_A = "lunarlander_3seed_v2_skeleton_diag"
SEEDS_A = [42, 43, 44]

def fig_a1_scores():
    """A1: v2 final score per seed"""
    scores = [load_best(EXP_A, s) for s in SEEDS_A]
    fig, ax = plt.subplots(figsize=(6, 4.5))
    colors = [GREEN if v >= SOLVED else RED for v in scores]
    bars = ax.bar([f"Seed {s}" for s in SEEDS_A], scores, color=colors, width=0.5)
    ax.axhline(SOLVED, color=RED, ls='--', lw=1.5, label=f'Solved ({SOLVED})')
    ax.axhline(BASELINE, color=GRAY, ls=':', lw=1.5, label=f'PPO Baseline ({BASELINE:.0f})')
    for b, v in zip(bars, scores):
        ax.text(b.get_x()+b.get_width()/2, v+5, f'{v:.0f}', ha='center', fontsize=13, fontweight='bold')
    ax.set_ylabel('Original Environment Return')
    ax.set_title('v2 3-Seed: Final Best Score per Seed')
    ax.legend(frameon=False, loc='lower right')
    ax.set_ylim(0, max(scores)+40)
    fig.tight_layout(); fig.savefig(OUT/"A1_v2_scores.png", bbox_inches='tight'); plt.close(fig)

def fig_a2_trajectories():
    """A2: v2 iteration trajectory all 3 seeds"""
    fig, ax = plt.subplots(figsize=(8, 5))
    for s in SEEDS_A:
        d = load(EXP_A, s)
        scores = [r['score'] for r in d]
        ax.plot(range(len(scores)), scores, marker='o', lw=2, label=f'Seed {s}')
    ax.axhline(SOLVED, color=RED, ls='--', lw=1.5)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Score')
    ax.set_title('v2 3-Seed: Iteration Score Trajectories')
    ax.legend(frameon=False); ax.set_ylim(-50, 280)
    fig.tight_layout(); fig.savefig(OUT/"A2_v2_trajectories.png", bbox_inches='tight'); plt.close(fig)

def fig_a3_convergence():
    """A3: v2 iterations to solve"""
    iters_to_solve = []
    for s in SEEDS_A:
        d = load(EXP_A, s)
        for r in d:
            if r['score'] >= SOLVED:
                iters_to_solve.append(r['iteration'])
                break
    fig, ax = plt.subplots(figsize=(5.5, 4))
    ax.bar([f"Seed {s}" for s in SEEDS_A], iters_to_solve, color=GREEN, width=0.5)
    ax.set_ylabel('Iterations to Solve')
    ax.set_title('v2 3-Seed: Convergence Speed')
    for i, (b, v) in enumerate(zip(SEEDS_A, iters_to_solve)):
        ax.text(i, v+0.15, f'{v}', ha='center', fontsize=14, fontweight='bold')
    fig.tight_layout(); fig.savefig(OUT/"A3_v2_convergence.png", bbox_inches='tight'); plt.close(fig)

def fig_a4_detail():
    """A4: v2 per-seed score + episode length"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, s in zip(axes, SEEDS_A):
        d = load(EXP_A, s)
        scores = [r['score'] for r in d]
        lengths = [r['mean_episode_length'] for r in d]
        ax2 = ax.twinx()
        ax.plot(range(len(scores)), scores, 'o-', color=BLUE, lw=2, ms=7, label='Score')
        ax2.plot(range(len(lengths)), lengths, 's--', color=PURPLE, lw=1.5, ms=5, label='Ep Length')
        ax.axhline(SOLVED, color=RED, ls='--', lw=1)
        ax.set_title(f'Seed {s}'); ax.set_xlabel('Iteration')
        ax.set_ylabel('Score', color=BLUE); ax2.set_ylabel('Length', color=PURPLE)
    fig.suptitle('v2 3-Seed: Score & Episode Length per Iteration', fontsize=15, y=1.02)
    fig.tight_layout(); fig.savefig(OUT/"A4_v2_detail.png", bbox_inches='tight'); plt.close(fig)

# ═══════════════════════════════════════════════════
#  SET B: 10-seed final experiment
# ═══════════════════════════════════════════════════
EXP_B = "lunarlander_10seed_1M_final"
SEEDS_B = list(range(42, 52))

def fig_b1_scores():
    """B1: 10-seed score distribution"""
    best = {s: load_best(EXP_B, s) for s in SEEDS_B if load_best(EXP_B, s) is not None}
    solved_n = sum(1 for v in best.values() if v >= SOLVED)
    colors = [GREEN if v >= SOLVED else RED for v in best.values()]
    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar([f"{s}" for s in best], list(best.values()), color=colors, width=0.6)
    ax.axhline(SOLVED, color=RED, ls='--', lw=1.5, label=f'Solved ({SOLVED})')
    for b, v in zip(bars, best.values()):
        y = v+8 if v>=0 else v-18
        ax.text(b.get_x()+b.get_width()/2, y, f'{v:.0f}', ha='center', fontsize=8.5)
    ax.set_xlabel('Seed'); ax.set_ylabel('Best Score')
    ax.set_title(f'10-Seed Independent: Final Scores ({solved_n}/{len(best)} Solved)')
    ax.legend(frameon=False)
    fig.tight_layout(); fig.savefig(OUT/"B1_10seed_scores.png", bbox_inches='tight'); plt.close(fig)

def fig_b2_trajectories():
    """B2: 10-seed all trajectories (solved in green, unsolved in red)"""
    fig, ax = plt.subplots(figsize=(12, 6))
    for s in SEEDS_B:
        d = load(EXP_B, s)
        if not d: continue
        scores = [r['score'] for r in d]
        best_v = max(scores)
        color = GREEN if best_v >= SOLVED else RED
        alpha = 0.9 if best_v >= SOLVED else 0.4
        lw = 2 if best_v >= SOLVED else 1
        ax.plot(range(len(scores)), scores, color=color, alpha=alpha, lw=lw, marker='o' if best_v >= SOLVED else '.', ms=5 if best_v >= SOLVED else 3, label=f'Seed {s}' if best_v >= SOLVED else '')
    ax.axhline(SOLVED, color=RED, ls='--', lw=1.5)
    ax.set_xlabel('Iteration'); ax.set_ylabel('Score')
    ax.set_title('10-Seed: All Iteration Trajectories (green=solved, red=unsolved)')
    handles = [plt.Line2D([0],[0],color=GREEN,lw=2,label=f'Solved ({sum(1 for s in SEEDS_B if load_best(EXP_B,s) and load_best(EXP_B,s)>=SOLVED)})'),
               plt.Line2D([0],[0],color=RED,lw=1,label=f'Unsolved ({sum(1 for s in SEEDS_B if load_best(EXP_B,s) and load_best(EXP_B,s)<SOLVED)})')]
    ax.legend(handles=handles, frameon=False)
    fig.tight_layout(); fig.savefig(OUT/"B2_10seed_trajectories.png", bbox_inches='tight'); plt.close(fig)

def fig_b3_success_rate():
    """B3: Success rate pie"""
    best = [load_best(EXP_B, s) for s in SEEDS_B if load_best(EXP_B, s) is not None]
    s_n = sum(1 for v in best if v >= SOLVED)
    u_n = len(best) - s_n
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.pie([s_n, u_n], labels=[f'Solved ({s_n})', f'Unsolved ({u_n})'],
           colors=[GREEN, RED], autopct='%1.0f%%', startangle=90,
           textprops={'fontsize': 14, 'fontweight': 'bold'})
    ax.set_title(f'10-Seed Success Rate: {s_n}/{len(best)} ({100*s_n/len(best):.0f}%)')
    fig.tight_layout(); fig.savefig(OUT/"B3_10seed_success_rate.png", bbox_inches='tight'); plt.close(fig)

def fig_b4_solved_detail():
    """B4: Solved seeds iteration detail"""
    solved_seeds = [s for s in SEEDS_B if load_best(EXP_B, s) and load_best(EXP_B, s) >= SOLVED]
    n = len(solved_seeds)
    if n == 0: return
    fig, axes = plt.subplots(1, n, figsize=(4*n, 4))
    if n == 1: axes = [axes]
    for ax, s in zip(axes, solved_seeds):
        d = load(EXP_B, s)
        scores = [r['score'] for r in d]
        ax.plot(range(len(scores)), scores, 'o-', color=GREEN, lw=2, ms=7)
        ax.axhline(SOLVED, color=RED, ls='--', lw=1)
        ax.set_title(f'Seed {s}'); ax.set_xlabel('Iteration'); ax.set_ylabel('Score')
    fig.suptitle('10-Seed: Solved Seeds Individual Trajectories', fontsize=15, y=1.02)
    fig.tight_layout(); fig.savefig(OUT/"B4_10seed_solved_detail.png", bbox_inches='tight'); plt.close(fig)

def fig_b5_unsolved_detail():
    """B5: Unsolved seeds iteration detail"""
    unsolved = [s for s in SEEDS_B if load_best(EXP_B, s) and load_best(EXP_B, s) < SOLVED]
    n = len(unsolved)
    if n == 0: return
    cols = min(3, n)
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5*cols, 3.5*rows))
    axes = axes.flatten() if n > 1 else [axes]
    for ax, s in zip(axes, unsolved):
        d = load(EXP_B, s)
        scores = [r['score'] for r in d]
        ax.plot(range(len(scores)), scores, 'o-', color=RED, lw=1.5, ms=5)
        ax.axhline(0, color=GRAY, ls='-', lw=0.5)
        ax.set_title(f'Seed {s} (best={max(scores):.0f})'); ax.set_xlabel('Iteration'); ax.set_ylabel('Score')
    for ax in axes[len(unsolved):]: ax.set_visible(False)
    fig.suptitle('10-Seed: Unsolved Seeds (stuck or oscillating)', fontsize=15, y=1.02)
    fig.tight_layout(); fig.savefig(OUT/"B5_10seed_unsolved_detail.png", bbox_inches='tight'); plt.close(fig)

def fig_b6_convergence():
    """B6: Distribution of iterations to solve"""
    iters = []
    for s in SEEDS_B:
        d = load(EXP_B, s)
        if not d: continue
        for r in d:
            if r['score'] >= SOLVED:
                iters.append(r['iteration'])
                break
    if not iters: return
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.hist(iters, bins=range(0, max(iters)+2, 2), color=GREEN, edgecolor='white', alpha=0.8)
    ax.set_xlabel('Iterations to Solve'); ax.set_ylabel('Count')
    ax.set_title(f'Convergence Speed (median={np.median(iters):.0f}, mean={np.mean(iters):.1f})')
    fig.tight_layout(); fig.savefig(OUT/"B6_10seed_convergence.png", bbox_inches='tight'); plt.close(fig)

# ═══════════════════════════════════════════════════
#  SET C: Cross-experiment comparison
# ═══════════════════════════════════════════════════
def fig_c1_evidence_chain():
    """C1: Evidence chain"""
    v2_scores = [load_best(EXP_A, s) for s in SEEDS_A]
    v2_mean = np.mean(v2_scores)
    b_solved = [load_best(EXP_B, s) for s in SEEDS_B if load_best(EXP_B, s) and load_best(EXP_B, s) >= SOLVED]
    b_mean_solved = np.mean(b_solved) if b_solved else 0

    fig, ax = plt.subplots(figsize=(7, 5))
    labels = ['PPO Baseline\n(Official Reward)', 'FDRE-HRDC v2\n(3 seeds, 3/3 solved)', 'FDRE-HRDC 10-Seed\n(5/10 solved, solved mean)']
    values = [BASELINE, v2_mean, b_mean_solved]
    colors = [GRAY, BLUE, PURPLE]
    bars = ax.bar(labels, values, color=colors, width=0.5)
    ax.axhline(SOLVED, color=RED, ls='--', lw=1.5)
    for b, v in zip(bars, values):
        ax.text(b.get_x()+b.get_width()/2, v+5, f'{v:.0f}', ha='center', fontsize=15, fontweight='bold')
    ax.set_ylabel('Mean Original Environment Return')
    ax.set_title('Evidence Chain: Baseline → Agent Reward Search')
    fig.tight_layout(); fig.savefig(OUT/"C1_evidence_chain.png", bbox_inches='tight'); plt.close(fig)

def fig_c2_iteration_comparison():
    """C2: v2 vs 10-seed score distribution side-by-side"""
    v2_best = [load_best(EXP_A, s) for s in SEEDS_A]
    b_best = [load_best(EXP_B, s) for s in SEEDS_B if load_best(EXP_B, s) is not None]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # v2
    colors1 = [GREEN if v >= SOLVED else RED for v in v2_best]
    ax1.bar([f'S{s}' for s in SEEDS_A], v2_best, color=colors1, width=0.5)
    ax1.axhline(SOLVED, color=RED, ls='--', lw=1)
    ax1.set_title(f'v2 (3/3 solved)'); ax1.set_ylabel('Score')

    # 10-seed
    colors2 = [GREEN if v >= SOLVED else RED for v in b_best]
    ax2.bar([f'S{s}' for s in SEEDS_B], b_best, color=colors2, width=0.5)
    ax2.axhline(SOLVED, color=RED, ls='--', lw=1)
    ax2.set_title(f'10-Seed ({sum(1 for v in b_best if v>=SOLVED)}/10 solved)')

    fig.suptitle('Side-by-Side: v2 vs 10-Seed Score Distribution', fontsize=15, y=1.02)
    fig.tight_layout(); fig.savefig(OUT/"C2_comparison.png", bbox_inches='tight'); plt.close(fig)

# ═══════════════════════════════════════════════════

if __name__ == "__main__":
    # Set A
    fig_a1_scores()
    fig_a2_trajectories()
    fig_a3_convergence()
    fig_a4_detail()
    # Set B
    fig_b1_scores()
    fig_b2_trajectories()
    fig_b3_success_rate()
    fig_b4_solved_detail()
    fig_b5_unsolved_detail()
    fig_b6_convergence()
    # Set C
    fig_c1_evidence_chain()
    fig_c2_iteration_comparison()

    pngs = sorted(OUT.glob("*.png"))
    print(f"Generated {len(pngs)} figures:")
    for p in pngs:
        print(f"  {p.name}")
