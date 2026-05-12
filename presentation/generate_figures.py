"""Generate all 6 presentation figures from dev-log derived data.

Run from project root:
    python -m presentation.generate_figures

Outputs: presentation/figures/0[1-6]_*.png
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyBboxPatch
from matplotlib.lines import Line2D

OUT = Path(__file__).parent / "figures"
OUT.mkdir(parents=True, exist_ok=True)

PALETTE = {
    "bg":      "#0f1419",
    "panel":   "#161b22",
    "grid":    "#30363d",
    "text":    "#e6edf3",
    "muted":   "#8b949e",
    "blue":    "#58a6ff",
    "cyan":    "#79c0ff",
    "purple":  "#d2a8ff",
    "green":   "#56d364",
    "orange":  "#ffa657",
    "red":     "#f85149",
    "yellow":  "#e3b341",
}

plt.rcParams.update({
    "figure.facecolor":  PALETTE["bg"],
    "axes.facecolor":    PALETTE["panel"],
    "axes.edgecolor":    PALETTE["grid"],
    "axes.labelcolor":   PALETTE["text"],
    "axes.titlecolor":   PALETTE["cyan"],
    "xtick.color":       PALETTE["muted"],
    "ytick.color":       PALETTE["muted"],
    "text.color":        PALETTE["text"],
    "grid.color":        PALETTE["grid"],
    "grid.alpha":        0.4,
    "font.family":       "DejaVu Sans",
    "font.size":         11,
    "axes.titlesize":    14,
    "axes.titleweight":  "bold",
    "legend.facecolor":  PALETTE["panel"],
    "legend.edgecolor":  PALETTE["grid"],
    "legend.labelcolor": PALETTE["text"],
    "savefig.facecolor": PALETTE["bg"],
    "savefig.dpi":       150,
})


def fig_01_pipeline_overview() -> None:
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 12)
    ax.axis("off")

    phases = [
        ("Ph.0  INDI Hover Gate",      "tilt 0.00deg, omega 0",                    PALETTE["green"],  10.5),
        ("Ph.1  CTBR PPO Expert",      "RMSE 0.065m, 0/50 crash",                  PALETTE["green"],  9.0),
        ("Ph.2  FPV Data Collection",  "1000 ep, 500k steps, 3.9 GB",              PALETTE["green"],  7.5),
        ("Ph.3a  Flow Matching BC",    "val 0.063, eval 0.522m, 50/50 crash",      PALETTE["green"],  6.0),
        ("Ph.3b  ReinFlow RL Fine-tune", "20 runs, best 0.300m, 50/50 crash",      PALETTE["orange"], 4.5),
        ("Ph.4  Jetson Deploy",        "future, 122 Hz target met",                PALETTE["red"],    3.0),
    ]
    status = {
        PALETTE["green"]:  "[Done]",
        PALETTE["orange"]: "[20 runs]",
        PALETTE["red"]:    "[Planned]",
    }

    for title, sub, color, y in phases:
        box = FancyBboxPatch(
            (0.5, y - 0.55), 9, 1.1,
            boxstyle="round,pad=0.06",
            edgecolor=color, facecolor=PALETTE["panel"],
            linewidth=2.2,
        )
        ax.add_patch(box)
        ax.text(0.85, y + 0.18, title, fontsize=12.2, fontweight="bold", color=color)
        ax.text(0.85, y - 0.30, sub, fontsize=10.0, color=PALETTE["muted"])
        ax.text(9.2, y, status[color], fontsize=9.5, color=color, ha="right", va="center")

    for y in [9.7, 8.2, 6.7, 5.2, 3.7]:
        ax.annotate("", xy=(5, y - 0.05), xytext=(5, y + 0.1),
                    arrowprops=dict(arrowstyle="->", color=PALETTE["grid"], lw=1.5))

    ax.text(5, 11.7, "Research Pipeline (v4.0)",
            fontsize=15, fontweight="bold", color=PALETTE["cyan"], ha="center")
    ax.text(5, 1.6,
            "Best eval today:  ReinFlow Run 10  -  RMSE 0.300m  -  still 50/50 crash",
            fontsize=11, color=PALETTE["yellow"], ha="center", style="italic")
    ax.text(5, 0.9,
            "Surprise baseline:  PID Cascade  -  RMSE 0.022m  -  0/50 crash (hover)",
            fontsize=11, color=PALETTE["green"], ha="center", style="italic")

    fig.tight_layout()
    fig.savefig(OUT / "01_pipeline_overview.png", bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def fig_02_progress_timeline() -> None:
    fig, ax = plt.subplots(figsize=(10, 5.2))
    events = [
        ("04-01", 1.0, "PPO v3 Expert",       PALETTE["blue"]),
        ("04-12", 1.0, "v3.3 supervised",     PALETTE["blue"]),
        ("04-13", 1.0, "v3.3 DPPO Run 1\n0.104m",  PALETTE["purple"]),
        ("04-19", 1.0, "v4 PPO Expert\n0.065m",   PALETTE["green"]),
        ("04-20", 1.0, "v4 BC pretrain\nval 0.063", PALETTE["green"]),
        ("04-20", 0.0, "ReinFlow Run 1",     PALETTE["orange"]),
        ("04-27", 0.0, "Run 10\n0.300m",      PALETTE["yellow"]),
        ("04-29", 0.0, "Run 12\n0.297m",      PALETTE["yellow"]),
        ("05-02", 0.0, "Run 19\nreward 0.695", PALETTE["yellow"]),
        ("05-03", 0.0, "Run 20",             PALETTE["orange"]),
        ("05-06", -1.0, "PID Baseline\n0.022m", PALETTE["green"]),
        ("05-06", -1.0, "Temp ablation\n(neg result)", PALETTE["red"]),
    ]

    x = np.arange(len(events))
    for i, (date, lane, label, color) in enumerate(events):
        ax.scatter(i, lane, s=200, color=color, zorder=3, edgecolor=PALETTE["bg"], linewidth=2)
        offset = 0.45 if i % 2 == 0 else -0.7
        ax.annotate(label, (i, lane), xytext=(i, lane + offset),
                    fontsize=9, ha="center", color=color,
                    arrowprops=dict(arrowstyle="-", color=color, lw=0.8, alpha=0.6))
        ax.text(i, -2.2, date, fontsize=8, color=PALETTE["muted"],
                ha="center", rotation=30)

    for lane, label in [(1.0, "Foundation"), (0.0, "ReinFlow RL"), (-1.0, "Diagnostics")]:
        ax.axhline(lane, color=PALETTE["grid"], lw=1, alpha=0.5, zorder=1)
        ax.text(-0.7, lane, label, fontsize=10, color=PALETTE["cyan"],
                va="center", ha="right", fontweight="bold")

    ax.set_xlim(-1.5, len(events) - 0.5)
    ax.set_ylim(-2.6, 1.9)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_title("Progress Timeline   (2026-04-01 -> 2026-05-06)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "02_progress_timeline.png", bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def fig_03_method_comparison() -> None:
    methods = [
        ("PID\nHover",       0.022,   0,  PALETTE["green"]),
        ("PPO Expert\nHover", 0.065,  0,  PALETTE["blue"]),
        ("PID\nWaypoint",    1.177,   0,  PALETTE["yellow"]),
        ("BC\nWaypoint",     0.522,  50,  PALETTE["red"]),
        ("Run 19\nWaypoint", 0.523,  50,  PALETTE["orange"]),
        ("Run 12\nWaypoint", 0.297,  50,  PALETTE["purple"]),
        ("Run 10\nWaypoint", 0.300,  50,  PALETTE["cyan"]),
        ("v3.3 DPPO\nWaypoint", 0.104, 50, PALETTE["purple"]),
    ]
    labels   = [m[0] for m in methods]
    rmse     = [m[1] for m in methods]
    crashes  = [m[2] for m in methods]
    colors   = [m[3] for m in methods]
    x = np.arange(len(methods))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    bars = ax.bar(x, rmse, color=colors, edgecolor=PALETTE["bg"], linewidth=1.5)
    for b, v in zip(bars, rmse):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.03, f"{v:.3f}",
                ha="center", fontsize=9, color=PALETTE["text"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Position RMSE (m)")
    ax.set_title("Position RMSE  (lower is better)")
    ax.axhline(0.15, color=PALETTE["green"], lw=1.2, ls="--", alpha=0.7)
    ax.text(7.4, 0.16, "Gate 0.15m", color=PALETTE["green"], fontsize=8, ha="right")
    ax.set_ylim(0, 1.35)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    bars = ax.bar(x, crashes, color=colors, edgecolor=PALETTE["bg"], linewidth=1.5)
    for b, v in zip(bars, crashes):
        label = f"{v}/50"
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, label,
                ha="center", fontsize=9, color=PALETTE["text"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Crashes / 50 ep")
    ax.set_title("Crash Count  (lower is better)")
    ax.set_ylim(0, 60)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Method Comparison  -  the 50/50 wall is universal across all RL variants",
                 color=PALETTE["yellow"], fontsize=13, y=1.02)
    fig.tight_layout()
    fig.savefig(OUT / "03_method_comparison.png", bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def fig_04_run_history() -> None:
    runs = list(range(1, 21))
    train_peak = [0.668, 0.646, 0.525, 0.641, 0.645, 0.648, 0.649, 0.472,
                  0.534, 0.658, 0.312, 0.827, 0.820, 0.820, 0.820, 0.820,
                  0.760, 0.700, 0.6948, 0.6948]
    eval_rmse  = [0.522, 0.513, np.nan, np.nan, np.nan, np.nan, 0.514, np.nan,
                  0.516, 0.300, 0.142, 0.297, 0.510, 0.510, 0.510, 0.510,
                  np.nan, 0.510, 0.523, np.nan]

    notes = {
        3:  "PLoss=0",
        5:  "VLoss osc.",
        8:  "gate closed",
        10: "first real eval drop",
        11: "RMSE artefact",
        12: "soft penalty",
        18: "VLoss spike",
        19: "LR=1e-7 unlocks ceiling",
    }

    fig, ax1 = plt.subplots(figsize=(11, 5.5))
    ax2 = ax1.twinx()

    ax1.bar(runs, train_peak, color=PALETTE["blue"], alpha=0.65,
            label="Train peak reward", edgecolor=PALETTE["bg"], linewidth=1.0)

    valid = ~np.isnan(eval_rmse)
    ax2.plot(np.array(runs)[valid], np.array(eval_rmse)[valid],
             color=PALETTE["orange"], marker="o", markersize=8, linewidth=2.2,
             label="Eval RMSE (m)", zorder=4)
    for r, v in zip(runs, eval_rmse):
        if not np.isnan(v):
            ax2.text(r, v + 0.02, f"{v:.3f}", fontsize=8,
                     color=PALETTE["orange"], ha="center", fontweight="bold")

    for r, txt in notes.items():
        ax1.annotate(txt, (r, train_peak[r - 1]),
                     xytext=(r, train_peak[r - 1] + 0.18),
                     fontsize=8.5, ha="center", color=PALETTE["yellow"],
                     arrowprops=dict(arrowstyle="-", color=PALETTE["yellow"], lw=0.6, alpha=0.6))

    ax1.axhline(0.6948, color=PALETTE["purple"], lw=1.2, ls=":", alpha=0.8)
    ax1.text(20.3, 0.6948, "ceiling 0.6948", color=PALETTE["purple"], fontsize=8, va="center")
    ax2.axhline(0.522, color=PALETTE["red"], lw=1.2, ls=":", alpha=0.6)
    ax2.text(20.3, 0.522, "BC 0.522m", color=PALETTE["red"], fontsize=8, va="center")

    ax1.set_xlabel("Run #")
    ax1.set_ylabel("Train peak reward (per-step)", color=PALETTE["blue"])
    ax2.set_ylabel("Eval RMSE (m)", color=PALETTE["orange"])
    ax1.set_xticks(runs)
    ax1.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["orange"])
    ax1.set_ylim(0, 1.05)
    ax2.set_ylim(0, 0.65)
    ax1.set_title("ReinFlow 20-run history  -  reward climbs, eval RMSE doesn't")
    ax1.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUT / "04_run_history.png", bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def fig_05_train_eval_gap() -> None:
    runs   = ["BC", "Run 7", "Run 12", "Run 13", "Run 19", "Run 20"]
    reward = [0.529, 0.649, 0.827, 0.820, 0.6948, 0.6948]
    rmse   = [0.522, 0.514, 0.297, 0.510, 0.523, 0.523]
    crash_step = [60, 57, 22, 60, 61, 61]

    x = np.arange(len(runs))
    width = 0.35

    fig, axes = plt.subplots(2, 1, figsize=(9, 6.2), sharex=True,
                              gridspec_kw={"height_ratios": [3, 2]})

    ax = axes[0]
    b1 = ax.bar(x - width / 2, reward, width, color=PALETTE["blue"], label="Train reward")
    ax2 = ax.twinx()
    b2 = ax2.bar(x + width / 2, rmse, width, color=PALETTE["orange"], label="Eval RMSE")
    ax.set_ylabel("Train reward (per-step)", color=PALETTE["blue"])
    ax2.set_ylabel("Eval RMSE (m)", color=PALETTE["orange"])
    ax.tick_params(axis="y", labelcolor=PALETTE["blue"])
    ax2.tick_params(axis="y", labelcolor=PALETTE["orange"])

    for b, v in zip(b1, reward):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}",
                ha="center", fontsize=8.5, color=PALETTE["blue"], fontweight="bold")
    for b, v in zip(b2, rmse):
        ax2.text(b.get_x() + b.get_width() / 2, v + 0.015, f"{v:.3f}",
                 ha="center", fontsize=8.5, color=PALETTE["orange"], fontweight="bold")

    ax.set_title("Training-Eval Gap  -  train reward up 31%, eval RMSE flat")
    ax.set_ylim(0, 1.0)
    ax2.set_ylim(0, 0.65)

    ax = axes[1]
    bars = ax.bar(x, crash_step, color=PALETTE["red"],
                  edgecolor=PALETTE["bg"], linewidth=1.2)
    ax.axhspan(55, 67, color=PALETTE["yellow"], alpha=0.12,
               label="typical crash window 55-67")
    for b, v in zip(bars, crash_step):
        suffix = "  (earlier crash, soft penalty)" if v < 30 else ""
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v}{suffix}",
                ha="center", fontsize=8.8, color=PALETTE["text"], fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(runs)
    ax.set_ylabel("Avg crash step (eval, all 50/50)")
    ax.set_ylim(0, 80)
    ax.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(OUT / "05_train_eval_gap.png", bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def fig_06_temperature_ablation() -> None:
    sigma   = [1.0, 0.7, 0.5, 0.3]
    rmse    = [0.5226, 0.5264, 0.5362, 0.5564]
    rmse_se = [0.0183, 0.0182, 0.0240, 0.0237]
    crashes = [50, 50, 50, 50]
    avg_crash = [61.5, 61.7, 62.9, 66.1]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.errorbar(sigma, rmse, yerr=rmse_se, fmt="o-",
                color=PALETTE["orange"], markersize=10,
                ecolor=PALETTE["red"], capsize=5, linewidth=2.2)
    for s, r in zip(sigma, rmse):
        ax.text(s, r + 0.005, f"{r:.4f}", ha="center", fontsize=9,
                color=PALETTE["text"], fontweight="bold")
    ax.invert_xaxis()
    ax.set_xlabel("Temperature sigma  (1.0 = training, lower = quieter)")
    ax.set_ylabel("Eval RMSE (m)")
    ax.set_title("Lower noise -> WORSE RMSE  (hypothesis rejected)")
    ax.grid(alpha=0.3)
    ax.set_ylim(0.50, 0.59)

    ax = axes[1]
    bars = ax.bar(np.arange(4), avg_crash, color=PALETTE["cyan"],
                  edgecolor=PALETTE["bg"], linewidth=1.2)
    for b, v in zip(bars, avg_crash):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}",
                ha="center", fontsize=10, color=PALETTE["text"], fontweight="bold")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels([f"sigma={s}" for s in sigma])
    ax.set_ylabel("Avg crash step")
    ax.set_title("Crash step shifts +4.6 steps (~75 ms)  -  negligible")
    ax.set_ylim(0, 80)
    for c, x in zip(crashes, np.arange(4)):
        ax.text(x, 5, f"{c}/50 crash", ha="center", fontsize=8, color=PALETTE["red"])

    fig.suptitle("Temperature Scaling Ablation  -  noise is NOT the cause of crashes",
                 color=PALETTE["yellow"], fontsize=12, y=1.03)
    fig.tight_layout()
    fig.savefig(OUT / "06_temperature_ablation.png", bbox_inches="tight", facecolor=PALETTE["bg"])
    plt.close(fig)


def main() -> None:
    print(f"Writing figures to {OUT}")
    fig_01_pipeline_overview();    print("  01_pipeline_overview.png")
    fig_02_progress_timeline();    print("  02_progress_timeline.png")
    fig_03_method_comparison();    print("  03_method_comparison.png")
    fig_04_run_history();          print("  04_run_history.png")
    fig_05_train_eval_gap();       print("  05_train_eval_gap.png")
    fig_06_temperature_ablation(); print("  06_temperature_ablation.png")
    print("Done.")


if __name__ == "__main__":
    main()
