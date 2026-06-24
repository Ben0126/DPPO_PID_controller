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

  Fig 3  single_seed_swing.png
         Per-seed Tier-1 / survival of the two P1 baselines (3 seeds each):
         PPO-from-pixels swings 0% <-> 47% Tier-1 on the training seed alone, so a
         single-seed leaderboard row is unsafe -> report across-seed mean +/- std.
         (Sec 4 P1 baselines.)

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

    fig.suptitle('Measurement: the FPV cannot encode metric range where the policy operates '
                 '(but see Fig. 4 — this is a fixable artifact, and range is not what gates precision)',
                 fontsize=10.5, y=1.04)
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'crosshair_distance_saturation.png')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {out}')


def _rangecue_agg():
    """Aggregate cond-IAE / survival / Tier-1 per range-cue arm across the 3 seeds,
    reading the four frozen-P0 artifacts (s0 + s12, clean + noised)."""
    files = ['p3b_rc_clean_frozen.json', 'p3b_rc_noised_frozen.json',
             'p3b_rc_clean_s12_frozen.json', 'p3b_rc_noised_s12_frozen.json']
    data = {}
    for f in files:
        d = load(f)
        res = d.get('results', d)   # frozen-P0 JSONs key labels at top level
        for lbl, v in res.items():
            if isinstance(v, dict) and 'iae_steady_cond' in v:
                data[lbl] = v

    def arm_of(lbl):
        base = lbl
        for suf in ('_s1', '_s2'):
            if lbl.endswith(suf):
                base = lbl[:-3]
        return 'control' if base == 'control' else base

    arms = {}
    for lbl, v in data.items():
        arms.setdefault(arm_of(lbl), []).append(v)
    out = {}
    for arm, vs in arms.items():
        iae = np.array([x['iae_steady_cond'] for x in vs], float)
        ncond = np.array([x['n_conditional'] for x in vs], float)
        tier1 = float(np.mean([x['tier1_pass_rate'] for x in vs])) * 100
        # "collapse": the arm barely flies to the 250-step threshold, so its
        # conditional cond-IAE is a short-survival artifact, not precision.
        collapsed = bool(tier1 < 15.0 or np.nanmean(ncond) < 8.0)
        out[arm] = {
            'iae_mean': float(np.nanmean(iae)) if not np.all(np.isnan(iae)) else np.nan,
            'iae_std': float(np.nanstd(iae)) if not np.all(np.isnan(iae)) else np.nan,
            'tier1_mean': tier1,
            'survive_mean': float(np.mean([x['survival_mean'] for x in vs])) * 100,
            'ncond_mean': float(np.nanmean(ncond)),
            'collapsed': collapsed,
            'n': len(vs),
        }
    return out


def fig4_sensing_ablation():
    """Panel A: higher-res gate — far-range R² for 3 resolutions x {saturating crosshair,
    perspective target}; the info loss is a target artifact, not the pixel count.
    Panel B: range-cue intervention — even the oracle metric range barely moves cond-IAE
    (vs the 0.068 m oracle), noise erases it, and the richer 3D cue collapses survival."""
    gate = load('p3b_higher_res_gate.json')['results']
    rc = _rangecue_agg()

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(11.0, 4.4),
                                   gridspec_kw={'width_ratios': [1.05, 1.25]})

    # --- Panel A: gate far R^2 ---
    res = [64, 128, 256]
    cross = [gate[f'{r}px/crosshair_prod']['r2_far_>=1.5m'] for r in res]
    persp = [gate[f'{r}px/perspective_aa']['r2_far_>=1.5m'] for r in res]
    x = np.arange(len(res)); w = 0.38
    bC = axA.bar(x - w/2, cross, w, color='#c0392b', edgecolor='k', alpha=0.85,
                 label='production crosshair (saturating)')
    bP = axA.bar(x + w/2, persp, w, color='#27ae60', edgecolor='k', alpha=0.85,
                 label='perspective target (non-saturating)')
    for bars in (bC, bP):
        for b in bars:
            axA.text(b.get_x() + b.get_width()/2, max(b.get_height(), 0) + 0.02,
                     f'{b.get_height():.2f}', ha='center', va='bottom', fontsize=9)
    axA.axhline(0, color='k', lw=0.8)
    axA.set_xticks(x); axA.set_xticklabels([f'{r}px' for r in res])
    axA.set_ylabel('R²  decode far range (≥1.5 m) from image')
    axA.set_ylim(-0.1, 0.7)
    axA.set_title('Gate: far-range info is a target artifact,\nnot the pixel count', fontsize=10.5)
    axA.legend(loc='upper left', fontsize=8.3, framealpha=0.9)

    # --- Panel B: range-cue intervention cond-IAE ---
    order = [('control', 'control\n(no cue)'),
             ('scalar_clean', 'scalar\nσ=0'),
             ('scalar_noised', 'scalar\nσ=0.15'),
             ('pos3d_noised', 'pos3d\nσ=0.15'),
             ('pos3d_clean', 'pos3d\nσ=0')]
    xs = np.arange(len(order))
    means = [rc[a]['iae_mean'] for a, _ in order]
    stds = [rc[a]['iae_std'] for a, _ in order]
    cols = ['#2c3e50', '#2c7fb8', '#7fb3d5', '#e67e22', '#c0392b']
    for xi, (a, _), m, sd, c in zip(xs, order, means, stds, cols):
        if rc[a]['collapsed']:   # pos3d_clean: short-survival artifact, cond-IAE unreliable
            axB.bar(xi, 3.2, 0.6, color=c, alpha=0.28, edgecolor=c, hatch='//', zorder=2)
            axB.text(xi, 1.7, f"survival\ncollapse\nTier-1 {rc[a]['tier1_mean']:.0f}%\n"
                              f"(n_cond≈{rc[a]['ncond_mean']:.0f}/30)",
                     ha='center', va='center', fontsize=8.0, color=c, fontweight='bold')
        else:
            axB.bar(xi, m, 0.6, yerr=sd, capsize=4, color=c, edgecolor='k',
                    alpha=0.88, zorder=2)
            axB.text(xi, m + sd + 0.06, f'{m:.2f}', ha='center', va='bottom',
                     fontsize=9.5, fontweight='bold')
    axB.axhline(0.068, color='#16a085', lw=2, ls='--', zorder=3)
    axB.text(len(order) - 0.5, 0.16, 'state oracle 0.068 m', color='#16a085',
             fontsize=9, ha='right', va='bottom')
    axB.set_xticks(xs); axB.set_xticklabels([l for _, l in order], fontsize=9)
    axB.set_ylabel('cond-IAE (m)  — closed-loop precision')
    axB.set_ylim(0, 3.6)
    axB.set_title('Intervention: even the oracle range cue does not move precision\n'
                  '(~36× oracle); noise erases it; richer cue collapses survival',
                  fontsize=10.0)

    fig.suptitle('Precision is coverage/teacher-competence-gated, not sensing-gated',
                 fontsize=12, y=1.03)
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'sensing_ablation.png')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {out}')


def fig3_single_seed_swing():
    """Single-seed unreliability: per-seed Tier-1 / survival spread of the two P1
    baselines (3 seeds each), with the across-seed mean +/- std. Numbers read
    straight from the seed-aggregate artifact (per_seed = fractions 0-1; the
    *_mean/_std fields are already in percent)."""
    agg = load('baselines_frozen_seeds_aggregate.json')
    models = [('PPO_from_pixels', 'PPO-from-pixels\n(end-to-end RL)', '#c0392b'),
              ('BC_vision_only', 'BC-vision-only\n(vision→action MLP)', '#2c7fb8')]

    fig, (axT, axS) = plt.subplots(1, 2, figsize=(10.0, 4.5), sharey=True)

    def panel(ax, per_key, mean_key, std_key, title):
        for xi, (mk, _label, col) in enumerate(models):
            d = agg[mk]
            seeds = sorted(d['per_seed'].keys(), key=int)
            vals = [d['per_seed'][s][per_key] * 100.0 for s in seeds]   # frac -> %
            jit = np.linspace(-0.14, 0.14, len(vals))
            ax.scatter([xi + j for j in jit], vals, s=78, color=col, alpha=0.55,
                       edgecolor='k', linewidth=0.6, zorder=3)
            for j, s, v in zip(jit, seeds, vals):
                ax.annotate(f's{s}', (xi + j, v), textcoords='offset points',
                            xytext=(0, 9), ha='center', fontsize=8, color=col)
            m, sd = d[mean_key], d[std_key]               # already in percent
            ax.errorbar(xi + 0.32, m, yerr=sd, marker='D', ms=10, color=col,
                        capsize=6, lw=2.2, zorder=4)
            ax.annotate(f'{m:.1f} ± {sd:.1f}', (xi + 0.32, m),
                        textcoords='offset points', xytext=(11, 0), va='center',
                        fontsize=9.5, fontweight='bold', color=col)
        ax.set_xticks(range(len(models)))
        ax.set_xticklabels([l for _, l, _ in models], fontsize=9)
        ax.set_xlim(-0.55, len(models) - 1 + 0.95)
        ax.set_ylim(-8, 108)
        ax.axhline(0, color='k', lw=0.6)
        ax.set_title(title, fontsize=11)

    panel(axT, 'tier1', 'tier1_mean', 'tier1_std', 'Tier-1 pass-rate')
    panel(axS, 'survival', 'survival_mean', 'survival_std', 'Survival')
    axT.set_ylabel('closed-loop % (per seed = dot, mean ± std = diamond)')

    # highlight the PPO-from-pixels Tier-1 swing (seeds 0 & 2 -> 0%, seed 1 -> ~47%)
    axT.annotate('', xy=(-0.33, 46.7), xytext=(-0.33, 0.0),
                 arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.6))
    axT.text(-0.26, 26.0, 'same model,\ndifferent seed:\n0% ↔ 47%', color='#c0392b',
             fontsize=8.5, ha='left', va='center', style='italic')

    fig.suptitle('One training seed is not a safe leaderboard row '
                 '(P1 baselines, 3 seeds each)', fontsize=12, y=1.0)
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'single_seed_swing.png')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {out}')


def fig5_ablation_forest():
    """2x2 ablation forest plot (faithful Dispersive, p2f): Tier-1 and survival per
    cell with across-seed mean +/- std. The decisive D1E1-vs-D0E1 contrast falls
    inside the pooled across-seed std -> Dispersive has no effect above seed noise."""
    lb = load('p2f_ablation_leaderboard.json')['cell_agg']

    # top-to-bottom: the decisive E2E pair first, then the frozen pair
    rows = [('D1E1', 'D1E1  ON / E2E', True),
            ('D0E1', 'D0E1  OFF / E2E', False),
            ('D1E0', 'D1E0  ON / frozen', True),
            ('D0E0', 'D0E0  OFF / frozen', False)]
    y = np.arange(len(rows))[::-1]

    def col(on):  return '#c0392b' if on else '#2c7fb8'
    def mark(on): return 's' if on else 'o'

    # pooled across-seed std for the decisive comparison (matches the paper's 6.3 pp)
    pooled = 100 * float(np.sqrt(lb['D1E1']['tier1_std']**2 + lb['D0E1']['tier1_std']**2))

    fig, (axT, axS) = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=True)

    def panel(ax, mean_key, std_key, title, band_cell=None):
        if band_cell is not None:
            c = lb[band_cell]['tier1_mean'] * 100
            ax.axvspan(c - pooled, c + pooled, color='#7f8c8d', alpha=0.14, zorder=1,
                       label=f'D0E1 ± pooled std ({pooled:.1f} pp)')
        for yi, (cell, _lab, on) in zip(y, rows):
            m = lb[cell][mean_key] * 100
            s = lb[cell][std_key] * 100
            ax.errorbar(m, yi, xerr=s, fmt=mark(on), ms=10, color=col(on),
                        capsize=5, lw=2, mec='k', mew=0.6, zorder=3)
            ax.annotate(f'{m:.1f}±{s:.1f}', (m, yi), textcoords='offset points',
                        xytext=(0, 11), ha='center', fontsize=8.6, color=col(on))
        ax.set_yticks(y); ax.set_yticklabels([r[1] for r in rows], fontsize=9)
        ax.set_ylim(-0.6, len(rows) - 0.4)
        ax.set_xlabel(title)
        ax.grid(axis='x', ls=':', alpha=0.5)

    panel(axT, 'tier1_mean', 'tier1_std', 'Tier-1 pass-rate (%)', band_cell='D0E1')
    panel(axS, 'survival_mean', 'survival_std', 'Survival (%)')

    # annotate the decisive null on the Tier-1 panel
    d = (lb['D1E1']['tier1_mean'] - lb['D0E1']['tier1_mean']) * 100
    axT.annotate(f'D1E1 − D0E1 = {d:+.1f} pp\n(∈ noise)', xy=(lb['D1E1']['tier1_mean']*100, y[0]),
                 xytext=(58, 3.0), fontsize=8.8, color='#7f0000', ha='center',
                 arrowprops=dict(arrowstyle='->', color='#7f0000', lw=1.1),
                 bbox=dict(facecolor='white', edgecolor='#7f0000', alpha=0.85, pad=1.5))
    axT.legend(loc='lower right', fontsize=8.0, framealpha=0.9)
    for ax in (axT, axS):
        ax.set_xlim(40, 100)

    fig.suptitle('Faithful Dispersive × E2E (2×2, 3 seeds): no Dispersive effect above seed noise',
                 fontsize=11.5, y=1.02)
    fig.tight_layout()
    out = os.path.join(FIGDIR, 'ablation_forest.png')
    fig.savefig(out, dpi=200, bbox_inches='tight'); plt.close(fig)
    print(f'wrote {out}')


def main():
    os.makedirs(FIGDIR, exist_ok=True)
    fig1_rank_survival()
    fig2_crosshair_saturation()
    fig3_single_seed_swing()
    fig4_sensing_ablation()
    fig5_ablation_forest()
    print(f'\nFigures in {FIGDIR}')


if __name__ == '__main__':
    main()
