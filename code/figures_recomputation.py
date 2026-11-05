#!/usr/bin/env python3
"""Figure 9 of study III (Section 4.7), English and Czech, PNG 300 dpi and PDF: the recomputation of Sections 4.3-4.5 by another principle.
Upper left: z of the 20 stratifications against random samples, originally (Section 4.4, open) and by layers (filled), against 1 - W100.
Upper right: reduction of layer 0 by the stratifications, exactly from the zeros against the line through two points.
Lower left: slope of the fitted weights of chi_D(p) against the explicit formula by cosine mode, diagonal terms alone (open) and with the dual term (filled).
Lower right: mean first zero minus USp(2N_eff(a)) with N_eff calibrated on [0, a], even (solid) and odd (dashed) characters.
Reads reference_recomputation.json (layered_recomputation.py), reference_explicit_weights.json (explicit_weights.py), reference_two_level.json
(two_level_first_zero.py) and reference_characters.json (character_sums.py) from DATA_FOLDER.
usage: python3 -I figures_recomputation.py DATA_FOLDER OUTPUT_FOLDER"""
import sys, json, math, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
DD, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
ld = lambda n: json.load(open(os.path.join(DD, n)))
H = ld('reference_recomputation.json'); EW = ld('reference_explicit_weights.json'); TL = ld('reference_two_level.json'); CH = ld('reference_characters.json')
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'axes.edgecolor': '#555555',
                     'axes.linewidth': 0.8, 'xtick.color': '#333333', 'ytick.color': '#333333', 'axes.grid': True, 'grid.color': '#e3e3e3', 'grid.linewidth': 0.6,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'legend.frameon': False})
INK, MUT = '#222222', '#6b6b6b'; COL = {'B10': '#c0392b', 'C100': '#1f5fa8', 'D100': '#2e8b57'}; FAM = ('B10', 'C100', 'D100'); ALLM = H['moduli']
CZF = FuncFormatter(lambda v, p: ('%g' % round(v, 6)).replace('.', ',').replace('-', '−'))
def comma(fig):
    for ax in fig.axes:
        if ax.get_yscale() == 'linear' and isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter): ax.yaxis.set_major_formatter(CZF)
        if ax.get_xscale() == 'linear' and isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter): ax.xaxis.set_major_formatter(CZF)
def save(fig, name):
    if name.endswith('_cz'): comma(fig)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None}); plt.close(fig)
for en in (False, True):
    sfx = '_en' if en else '_cz'; w = lambda c, e: e if en else c
    fig, A4 = plt.subplots(2, 2, figsize=(8.6, 6.8)); axs = [A4[0, 0], A4[0, 1], A4[1, 0], A4[1, 1]]
    # ---- (a) z of the stratifications, originally and by layers
    ax = axs[0]; dx = {'B10': -0.0025, 'C100': 0.0, 'D100': 0.0025}
    for f in FAM:
        F = H['families'][f]; O = F['original']['4.4']; Wt = CH['exact'][f]['types']; r0 = O['random']['low']
        xs = np.array([1 - Wt['mod%d' % M]['W100'] for M in ALLM]) + dx[f]
        zo = [(O['mod%d' % M]['low'][0] - r0[0])/math.hypot(O['mod%d' % M]['low'][1], r0[1]) for M in ALLM]; zl = [H['z_44'][f]['mod%d' % M] for M in ALLM]
        ax.plot(xs, zo, 'o', mfc='none', mec=COL[f], ms=4.5, mew=0.9); ax.plot(xs, zl, 'o', color=COL[f], ms=4.5, label=f)
    ax.axhline(0, color=MUT, lw=0.8); ax.axhline(-2, color=MUT, lw=0.6, ls=':')
    ax.set_xlabel(w('1 − W₁₀₀ (podíl rozptylu χ_D(p), p ≤ 100, který stratifikace odstraní)', '1 − W₁₀₀ (share of the variance of χ_D(p), p ≤ 100, removed)'))
    ax.set_ylabel(w('z proti náhodnému výběru', 'z against random samples')); ax.set_title(w('stratifikace: původně (prázdně) a po hladinách', 'stratifications: originally (open) and by layers'))
    ax.legend(loc='lower left')
    # ---- (b) reduction of layer 0: exact against the line through two points
    ax = axs[1]; mx = 0
    for f in FAM:
        rows = H['line_layer0_exact'][f]['rows']; rl = np.array([rows['mod%d' % M]['reduction_line'] for M in ALLM]); re_ = np.array([rows['mod%d' % M]['reduction_exact'] for M in ALLM])
        ax.plot(rl, re_, 'o', color=COL[f], ms=4.5, label='%s  %s %s' % (f, w('poměr', 'ratio'), ('%.2f' % H['line_layer0_exact'][f]['mean_ratio_of_reductions']).replace('.', ',' if not en else '.'))); mx = max(mx, rl.max(), re_.max())
    ax.plot([0, mx*1.05], [0, mx*1.05], '--', color=MUT, lw=0.9); ax.set_xlim(-0.005, mx*1.08); ax.set_ylim(-0.005, mx*1.08)
    ax.set_xlabel(w('snížení vrstvy 0 podle přímky dvou bodů', 'reduction of layer 0 by the line through two points')); ax.set_ylabel(w('snížení vrstvy 0 přesně z nul', 'reduction of layer 0 exactly from the zeros'))
    ax.set_title(w('přímka dvou bodů s přesnými hodnotami', 'line through two points with precise values')); ax.legend(loc='upper left')
    # ---- (c) explicit formula: slope by mode, diagonal and with the dual term
    ax = axs[2]; k = np.arange(1, 8)
    for i, f in enumerate(FAM):
        G = EW['families'][f]['regression']; sd = [a for a, b in G['slope_chi_by_mode']]; sf = [a for a, b in G['slope_chi_with_dual_by_mode']]
        ax.plot(k + (i - 1)*0.12, sd, 'o', mfc='none', mec=COL[f], ms=5, mew=0.9); ax.plot(k + (i - 1)*0.12, sf, 'o', color=COL[f], ms=5, label=f)
    ax.axhline(1, color=MUT, lw=0.9, ls='--'); ax.set_xticks(k); ax.set_ylim(-0.1, 1.25)
    ax.set_xlabel(w('kosinový mód k', 'cosine mode k')); ax.set_ylabel(w('sklon vah χ_D(p): fit / vzorec', 'slope of the weights of χ_D(p): fit / formula'))
    ax.set_title(w('explicitní formule: diagonální a s duálním', 'explicit formula: diagonal and with dual term')); ax.legend(loc='lower left')
    # ---- (d) first zero with N_eff calibrated on [0, a]
    ax = axs[3]
    for i, f in enumerate(FAM):
        for sg, ls, mk in (('plus', '-', 'o'), ('minus', '--', 's')):
            C = TL['families'][f][sg]['calibration']; a = np.array([q['a'] for q in C]) + (i - 1)*0.006; d = [q['difference'] for q in C]; e = [q['x1_se'] for q in C]
            ax.errorbar(a, d, yerr=e, fmt=mk + ls, color=COL[f], ms=3.5, lw=1.0, capsize=1.5, label=(f if sg == 'plus' else None))
    ax.axhline(0, color=MUT, lw=0.8); ax.axvline(0.5, color=MUT, lw=0.7, ls=':')
    ax.set_xlabel(w('konec intervalu kalibrace a (N_ef z hustoty na [0, a])', 'end of the calibration interval a (N_eff from the density on [0, a])'))
    ax.set_ylabel(w('průměr x₁ − USp(2N_ef(a))', 'mean x₁ − USp(2N_eff(a))')); ax.set_title(w('první nula: sudé (plně) a liché (čárkovaně)', 'first zero: even (solid) and odd (dashed)')); ax.legend(loc='upper right')
    for j, ax in enumerate(axs): ax.text(-0.12, 1.04, '(%s)' % 'abcd'[j], transform=ax.transAxes, fontsize=10, color=INK)
    fig.tight_layout(); save(fig, '09_recomputation' + sfx)
print('done')
