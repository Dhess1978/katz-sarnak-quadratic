#!/usr/bin/env python3
"""Figure 6 of study III (section 4.4; light background, English and Czech, PNG 300 dpi and PDF) from reference_characters.json (character_sums.py).
Left: the exact remaining share W100 of the weighted variance of the means of chi_D(p), p up to 100 (computed without zeros), against the low-mode power k = 1 ... 7 of the
zeros (computed without characters) for random samples, 20 stratifications and consecutive bands of B10, C100 and D100, with the line through two points.
Right: profile r_p of consecutive bands of 6 074 neighbouring discriminants against p.
usage: python3 -I figures_moduli.py reference_characters.json OUTPUT_FOLDER"""
import sys, json, math, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
RC, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True); C = json.load(open(RC)); SU = C['summary']; PR = C['profile']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'axes.edgecolor': '#555555',
                     'axes.linewidth': 0.8, 'xtick.color': '#333333', 'ytick.color': '#333333', 'axes.grid': True, 'grid.color': '#e3e3e3', 'grid.linewidth': 0.6,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'legend.frameon': False})
COL = {'B10': '#c0392b', 'C100': '#1f5fa8', 'D100': '#2e8b57'}; LAB = {'B10': '|D| 10⁴ … 10⁵', 'C100': '|D| 10⁶ … 2·10⁶', 'D100': '|D| ≈ 10⁸'}
CZF = FuncFormatter(lambda v, p: ('%g' % round(v, 6)).replace('.', ',').replace('-', '−'))
def save(fig, name):
    if name.endswith('_cz'):
        for ax in fig.axes:
            if ax.get_yscale() == 'linear' and isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter): ax.yaxis.set_major_formatter(CZF)
            if ax.get_xscale() == 'linear' and isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter): ax.xaxis.set_major_formatter(CZF)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None}); plt.close(fig)
for en in (False, True):
    sfx = '_en' if en else '_cz'; w = lambda c, e: e if en else c
    fig, axs = plt.subplots(1, 2, figsize=(7.6, 3.6))
    ax = axs[0]
    for jm in ('B10', 'C100', 'D100'):
        T = SU['families'][jm]['types']; r, s = T['random'], T['consecutive']; c = COL[jm]
        xs = np.linspace(s['W100'], 1.03, 50); ax.plot(xs, s['low'] + (r['low'] - s['low'])*(xs - s['W100'])/(1 - s['W100']), '--', color=c, lw=0.8)
        for key, t in T.items():
            if 'W100' not in t: continue
            x = 1.0 if key == 'random' else t['W100']; mk = 'o' if key not in ('random', 'consecutive') else ('s' if key == 'random' else 'D')
            ax.errorbar(x, t['low'], yerr=t['se'], fmt=mk, ms=3.2 if mk == 'o' else 4.5, color=c, mfc=c if mk != 'D' else 'white', mew=0.9, elinewidth=0.7, capsize=0)
        ax.plot([], [], 'o', color=c, ms=4, label=jm + ' (' + LAB[jm] + ')')
    ax.set_xlim(0, 1.06); ax.set_ylim(0, 0.062); ax.set_xlabel(w('$W_{100}$: zbylý rozptyl součtů $\\chi_D(p)$, $p \\leq 100$ (bez nul)', '$W_{100}$: remaining variance of the sums $\\chi_D(p)$, $p \\leq 100$ (no zeros)'))
    ax.set_ylabel(w('výkon nízkých módů k = 1 … 7 (nuly)', 'low-mode power k = 1 … 7 (zeros)')); X = SU['line']['W100_60']
    ax.set_title(w('20 stratifikací × pásma: χ² přímky (společné chyby) ', '20 stratifications × bands: χ² of the line (joint errors) ') + ('%.2f / %d' % (X['chi2'], X['n'])).replace('.', '.' if en else ','), fontsize=9)
    for mk, lab in (('D', w('souvislá pásma', 'consecutive bands')), ('s', w('náhodné výběry', 'random samples')), ('o', w('stratifikace D mod M', 'stratifications D mod M'))):
        ax.plot([], [], mk, color='#666666', mfc='white' if mk == 'D' else '#666666', ms=4.5 if mk != 'o' else 3.5, label=lab)
    ax.legend(loc='upper left', fontsize=7.5)
    ax = axs[1]; P = np.array(PR['primes'], float)
    for jm in ('B10', 'C100', 'D100'):
        r = np.array(PR['families'][jm]['r']); lx = np.log10(P); bins = np.linspace(np.log10(2), np.log10(5e6), 31); xc, yc = [], []
        for a, b in zip(bins[:-1], bins[1:]):
            m = (lx >= a) & (lx < b)
            if m.any(): xc.append(10**((a + b)/2)); yc.append(r[m].mean())
        ax.plot(xc, yc, '-o', ms=2.5, lw=1.2, color=COL[jm], label=jm); L = PR['families'][jm]['L']; ax.axvline(math.exp(7*L/8), color=COL[jm], lw=0.7, ls=':')
    ax.axvline(1e4, color='#555555', lw=0.9, ls='--'); ax.text(1.15e4, 1.28, 'Δ ≈ 10⁴', fontsize=7.5, color='#555555')
    ax.set_xscale('log'); ax.set_ylim(0, 1.4); ax.set_xlabel('p'); ax.set_ylabel(w('$r_p$ = rozptyl v pásmu / v náhodném výběru', '$r_p$ = variance in a band / in a random sample'))
    ax.set_title(w('Souvislá pásma po 6 074 D; tečkovaně $e^{7L/8}$', 'Consecutive bands of 6,074 D; dotted $e^{7L/8}$'), fontsize=9); ax.legend(loc='upper left', fontsize=7.5)
    fig.tight_layout(); save(fig, '06_moduli_characters' + sfx)
print('figure 6 written to', OUT)
