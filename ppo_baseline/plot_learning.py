"""
Learning curves for a PPO run, in the DiffRL trainers' layout.
==============================================================
rsl_rl logs to tensorboard; the DiffRL trainers write a PNG. This reads the
former and produces the latter, so a PPO curve and a DiffRL curve can sit side
by side in the thesis instead of being read from two different tools.

ON THE X AXIS
Iterations are NOT comparable between the two arms. A DiffRL iteration is one
BPTT pass over `batch` envs; a PPO iteration is `num_steps_per_env` transitions
over `num_envs`. At batch 20 vs 1500 envs those differ by orders of magnitude.
So the default x axis is ENVIRONMENT STEPS, which means the same thing on both
sides, and `--x iter` is available when comparing PPO runs with each other.

    python ppo_baseline/plot_learning.py ppo_runs/<run>
    python ppo_baseline/plot_learning.py ppo_runs/<run> --x iter
"""
import sys, os, argparse

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator


def smooth(y, k=9):
    if len(y) < k:
        return None
    pad = k // 2
    return np.convolve(np.r_[[y[0]] * pad, y, [y[-1]] * pad],
                       np.ones(k) / k, mode='valid')


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dir")
    p.add_argument("--out", default=None, help="default: <run_dir>/learning.png")
    p.add_argument("--x", choices=("steps", "iter"), default="steps")
    args = p.parse_args()

    ea = EventAccumulator(args.run_dir, size_guidance={'scalars': 0})
    ea.Reload()
    tags = set(ea.Tags()['scalars'])

    def get(tag):
        if tag not in tags:
            return None, None
        s = ea.Scalars(tag)
        return (np.array([x.step for x in s], float),
                np.array([x.value for x in s], float))

    # env-steps per iteration, recovered from the log itself rather than
    # assumed: total_fps * (collection+learning) time is the honest count.
    it, fps = get("Perf/total_fps")
    _, ct = get("Perf/collection_time")
    steps_per_iter = None
    if fps is not None and ct is not None:
        med = np.median((fps * ct)[1:]) if len(fps) > 1 else fps[0] * ct[0]
        steps_per_iter = float(med)

    def xs(steps):
        if args.x == "iter" or steps_per_iter is None:
            return steps, "Iteration"
        return steps * steps_per_iter, "Environment steps"

    seg_tags = sorted(t for t in tags if t.startswith("seg_frac/"))
    rows = 4 if seg_tags else 3
    fig, axes = plt.subplots(rows, 1, figsize=(10, 2.8 * rows), sharex=True)

    # ── reward ──
    ax = axes[0]
    st, v = get("Train/mean_reward")
    if v is not None:
        x, xlabel = xs(st)
        ax.plot(x, v, lw=1.0, color='steelblue', alpha=0.45, label='episode reward')
        sm = smooth(v)
        if sm is not None:
            ax.plot(x, sm, lw=2.0, color='navy', label='moving avg (9)')
        b = int(np.argmax(v))
        ax.plot(x[b], v[b], 'o', color='crimson', ms=5)
        ax.annotate(f"best {v[b]:.1f}", (x[b], v[b]), textcoords="offset points",
                    xytext=(-8, 6), ha='right', fontsize=8, color='crimson')
        ax.axhline(0.0, color='grey', lw=0.8, ls=':')
        ax.legend(fontsize=8, loc='upper left')
    ax.set_ylabel("Reward")
    ax.set_title("EPISODE REWARD", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ── success + curriculum ──
    ax = axes[1]
    st, v = get("Episode/success_rate")
    if v is not None:
        x, _ = xs(st)
        ax.plot(x, v * 100, lw=1.6, color='green', label='success %')
    ax.set_ylabel("Success %")
    ax.set_ylim(-5, 105)
    ax2 = ax.twinx()
    st, v = get("Episode/active_steps")
    if v is not None:
        x, _ = xs(st)
        ax2.plot(x, v, lw=1.4, color='darkorange', ls='--',
                 label='active steps (curriculum)')
        ax2.set_ylabel("active steps")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    ax.set_title("SUCCESS AND CURRICULUM PROGRESS", fontsize=10,
                 fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ── optimisation health ──
    ax = axes[2]
    st, v = get("Policy/mean_std")
    if v is not None:
        x, _ = xs(st)
        ax.plot(x, v, lw=1.4, color='purple', label='action sigma')
    ax.set_ylabel("sigma")
    ax3 = ax.twinx()
    st, v = get("Loss/value")
    if v is not None:
        x, _ = xs(st)
        ax3.semilogy(x, np.maximum(v, 1e-8), lw=1.2, color='brown',
                     alpha=0.8, label='value loss')
        ax3.set_ylabel("value loss")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax3.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper left')
    # sigma collapsing to the cap, or value loss to ~0, are the two failure
    # modes this run has already hit once each — worth seeing on every plot.
    ax.set_title("OPTIMISATION HEALTH  (sigma exploding, or value loss -> 0, "
                 "means no gradient signal)", fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.3)

    # ── per-segment pass fractions ──
    if seg_tags:
        ax = axes[3]
        cmap = plt.get_cmap('tab10')
        for i, t in enumerate(seg_tags):
            st, v = get(t)
            x, _ = xs(st)
            ax.plot(x, v * 100, lw=1.4, color=cmap(i % 10),
                    label=t.split('/')[-1])
        ax.axhline(50, color='crimson', lw=0.9, ls='--',
                   label='pass_frac 50%')
        ax.set_ylabel("envs passing %")
        ax.set_ylim(-5, 105)
        ax.legend(fontsize=7, loc='upper left', ncol=3)
        ax.set_title("SEGMENT PASS FRACTIONS  (a phase opens at 50% held for "
                     "`patience` ticks)", fontsize=10, fontweight='bold')
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel(xs(np.array([0]))[1])
    fig.suptitle(f"PPO — {os.path.basename(args.run_dir.rstrip('/'))}",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    if steps_per_iter:
        fig.text(0.995, 0.995, f"{steps_per_iter:,.0f} env-steps/iteration",
                 ha='right', va='top', fontsize=9, color='#333333',
                 bbox=dict(boxstyle='round,pad=0.3', facecolor='#f2f2f2',
                           edgecolor='#999999', lw=0.6))
    out = args.out or os.path.join(args.run_dir, "learning.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
