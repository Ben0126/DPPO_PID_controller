"""
Generate the two headline figures for the negative-result paper, reading numbers
straight from the frozen artifacts (no hand-transcription):

  Fig 1  rank_survival_decoupling.png
         `vis_pooled` effective rank swings ~15x across the 2x2 cells while
         closed-loop survival / Tier-1 stay flat -> representation collapse is not
         the binding constraint.  (Phase 3a + P2 leaderboard.)

  Fig 2  crosshair_distance_saturation.png
         The only FPV range cue (target crosshair size) saturates at 2 px beyond
         ~2 m, where the policy actually operates -> precision is information-gated.
         (Phase 3b image-distance-info + OOD coverage.)

Usage:
  dppo/Scripts/python.exe -m scripts.make_paper_figures
"""
import os
import sys
import json

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL = os.path.join(ROOT, 'evaluation_results')
FIGDIR = os.path.join(ROOT, 'docs', 'figures')


def load(name):
    with open(os.path.join(EVAL, name), 'r', encoding='utf-8') as f:
        return json.load(f)


def fig1_rank_survival():
    fc = load('p2_feature_collapse.json')['by_cell']
    lb = load('p2_ablation_leaderboard.json')['cell_agg']

    # Three distinct columns: frozen (D0E0==D1E0), D0E1, D1E1
    cols = [('D0E0', 'D0E0≡D1E0\n(frozen)'),
            ('D0E1', 'D0E1\n(E2E)'),
            ('D1E1', 'D1E1\n(E2E+Disp)')]
    rank = [fc[c]['effective_rank']['mean'] for c, _ in cols]
    surv = [lb[c]['survival_mean'] * 100 for c, _ in cols]
    surv_s = [lb[c]['survival_std'] * 100 for c, _ in cols]
    tier1 = [lb[c]['tier1_mean'] * 100 for c, _ in cols]
    tier1_s = [lb[c]['tier1_std'] * 100 for c, _ in cols]
    labels = [l for _, l in cols]
    x = np.arange(len(cols))

    fig, axL = plt.subplots(figsize=(7.0, 4.4))
    bars = axL.bar(x, rank, width=0.5, color='#b0c4de', edgecolor='#33506e',
                   zorder=2, label='vis_pooled effective rank')
    axL.set_yscale('log')
    axL.set_ylim(1, 60)
    axL.set_ylabel('effective rank of vis_pooled (log)', color='#33506e')
    axL.tick_params(axis='y', labelcolor='#33506e')
    axL.set_xticks(x); axL.set_xticklabels(labels)
    # rank value labels INSIDE each bar near the base (clear of all lines/markers)
    for xi, r in zip(x, rank):
        axL.text(xi, 1.18, f'{r:.1f}', ha='center', va='bottom',
                 fontsize=11, color='#1f3a57', fontweight='bold', zorder=6)

    axR = axL.twinx()
    axR.errorbar(x, surv, yerr=surv_s, marker='o', ms=8, lw=2.2, capsize=4,
                 color='#c0392b', label='survival %', zorder=3)
    axR.errorbar(x, tier1, yerr=tier1_s, marker='s', ms=7, lw=2.0, capsize=4,
                 ls='--', color='#e67e22', label='Tier-1 pass %', zorder=3)
    axR.set_ylim(0, 100)
    axR.set_ylabel('closed-loop %', color='#c0392b')
    axR.tick_params(axis='y', labelcolor='#c0392b')

    # annotate the decoupling
    axL.annotate('', xy=(1.9, 2.05), xytext=(0.12, 28.0),
                 arrowprops=dict(arrowstyle='->', color='#33506e', lw=1.4, ls=':'))
    axL.text(0.62, 16.5, '~15× rank drop', color='#33506e', fontsize=10,
             ha='center', style='italic', rotation=-26,
             bbox=dict(facecolor='white', edgecolor='none', alpha=0.7, pad=0.5))
    axR.text(1.5, 71.5, 'survival / Tier-1 flat', color='#c0392b', fontsize=10,
             ha='center', style='italic')

    lines = [bars] + axR.get_lines()
    axL.legend(lines, [b.get_label() for b in [bars]] +
               ['survival %', 'Tier-1 pass %'],
               loc='center left', fontsize=9, framealpha=0.9)
    axL.set_title('Representation collapse is decoupled from closed-loop survival',
                  fontsize=11.5)
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'rank_survival_decoupling.png')
    fig.savefig(out, dpi=200); plt.close(fig)
    print(f'wrote {out}')


def fig2_crosshair_saturation():
    on = load('p3b_image_distance_info.json')
    off = load('p3b_image_distance_info_nodr.json')
    cov = load('p3b_ood_coverage.json')

    ds = [float(d) for d in on['distances']]
    st_on = on['crosshair_size_by_distance']
    st_off = off['crosshair_size_by_distance']
    on_mean = [st_on[str(d)]['mean'] if str(d) in st_on else st_on[f'{d}']['mean'] for d in ds]

    def get(tbl, d, k):
        key = str(d) if str(d) in tbl else (f'{d}' if f'{d}' in tbl else None)
        # json keys may be '0.1' etc; build robustly
        for kk in tbl:
            if abs(float(kk) - d) < 1e-9:
                return tbl[kk][k]
        return np.nan
    on_mean = [get(st_on, d, 'mean') for d in ds]
    on_min = [get(st_on, d, 'min') for d in ds]
    on_max = [get(st_on, d, 'max') for d in ds]
    off_mean = [get(st_off, d, 'mean') for d in ds]

    r2 = on['ridge_r2_image_to_distance']
    steady = cov['closed_loop_steady']

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(10.5, 4.3),
                                   gridspec_kw={'width_ratios': [1.55, 1]})

    # --- Panel A: crosshair size vs distance ---
    axA.fill_between(ds, on_min, on_max, color='#c0392b', alpha=0.18,
                     label='DR-on size range (min–max)')
    axA.plot(ds, on_mean, '-o', color='#c0392b', lw=2, ms=5, label='DR-on size (mean)')
    axA.plot(ds, off_mean, '--s', color='#2c3e50', lw=1.8, ms=5,
             label='DR-off size (noiseless)')
    axA.axhline(2, color='gray', ls=':', lw=1.3)
    axA.text(2.95, 2.12, '2 px floor', color='gray', fontsize=9, ha='right')
    # saturated region
    axA.axvspan(2.0, max(ds), color='gray', alpha=0.10)
    axA.text(2.35, 4.7, 'saturated\n(no range info)', color='dimgray',
             fontsize=9, ha='center', style='italic')
    # policy operating point (steady-state drift): median + IQR
    axA.axvline(steady['p50'], color='#16a085', lw=2)
    axA.axvspan(steady['p50'], steady['p75'], color='#16a085', alpha=0.10)
    axA.text(steady['p50'] - 0.1, 3.3,
             f"policy drift\nmedian {steady['p50']:.1f} m", color='#16a085',
             fontsize=9, ha='right')
    axA.set_xlabel('target distance (m)')
    axA.set_ylabel('crosshair size (px)')
    axA.set_ylim(1.5, 6.5)
    axA.set_xlim(0, max(ds))
    axA.set_title('The only FPV range cue saturates at 2 m', fontsize=11)
    axA.legend(loc='upper right', fontsize=8.5, framealpha=0.9)

    # --- Panel B: distance decodability near vs far ---
    cats = ['near\n(<1 m)', 'far\n(≥1.5 m)']
    vals = [r2['near_<1m'], r2['far_>=1.5m']]
    colors = ['#27ae60', '#c0392b']
    bars = axB.bar(cats, vals, color=colors, width=0.55, edgecolor='k', alpha=0.85)
    for b, v in zip(bars, vals):
        axB.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.2f}',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')
    axB.set_ylim(0, 1.0)
    axB.set_ylabel('R²  (decode distance from image)')
    axB.set_title('Range is not recoverable far', fontsize=11)
    axB.axhline(0, color='k', lw=0.8)

    fig.suptitle('Precision is information-gated: the FPV cannot encode range where the policy operates',
                 fontsize=11.5, y=1.04)
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'crosshair_distance_saturation.png')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {out}')


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig1_rank_survival()
    fig2_crosshair_saturation()
    print(f'\nFigures in {FIGDIR}')


if __name__ == '__main__':
    main()
