#!/usr/bin/env python3
"""Figure 8 of study III (Section 4.6), English and Czech, PNG 300 dpi and PDF, from reference_layers.json of exact_layers.py:
left the low-mode power of subsets made of randomly placed pieces of l consecutive discriminants (layer 0 without draws as a line, the layered estimate
with its error, draws only for comparison, one stretch on the right), middle samples balanced on chi_D(p) for all p <= P (measured, and layer 0 from
the regression on chi_D(p)), lower left the correlation of the contributions of one L-function between the cosine modes in B10, lower right the increase
of the explained share of every mode by chi_D(n) of composite n (reference_intermodulation.json of intermodulation.py, next to reference_layers.json).
usage: python3 -I figures_layers.py reference_layers.json OUTPUT_FOLDER"""
import sys, json, math, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
RL, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True); J = json.load(open(RL)); FA = J['families']; N = J['subset_size']
IM = json.load(open(os.path.join(os.path.dirname(os.path.abspath(RL)), 'reference_intermodulation.json')))['families']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'axes.edgecolor': '#555555',
                     'axes.linewidth': 0.8, 'xtick.color': '#333333', 'ytick.color': '#333333', 'axes.grid': True, 'grid.color': '#e3e3e3', 'grid.linewidth': 0.6,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'legend.frameon': False})
INK, MUT = '#222222', '#6b6b6b'; COL = {'B10': '#c0392b', 'C100': '#1f5fa8', 'D100': '#2e8b57'}
def cz(s): return s.replace('.', ',')
def fmt(x, d, en): s = '%.*f' % (d, x); return s if en else cz(s)
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
    fig, A4 = plt.subplots(2, 2, figsize=(8.6, 6.6)); axs = [A4[0, 0], A4[0, 1], A4[1, 0], A4[1, 1]]
    ax = axs[0]
    for f in ('B10', 'C100', 'D100'):
        F = FA[f]; L = J['lengths']; D = F['designs']; c = COL[f]
        ax.plot(L + [N], [D['l%d' % l]['layer0'] for l in L] + [F['stretch']['layer0']], color=c, lw=1.2, label='%s (l = %s)' % (f, fmt(F['l'], 1, en)))
        ax.errorbar(L, [D['l%d' % l]['estimate'][0] for l in L], yerr=[D['l%d' % l]['estimate'][1] for l in L], fmt='o', ms=3.2, color=c, lw=0.8)
        ax.errorbar([l*1.12 for l in L], [D['l%d' % l]['draws_only'][0] for l in L], yerr=[D['l%d' % l]['draws_only'][1] for l in L], fmt='o', ms=3.0, mfc='white', color=c, lw=0.6)
        ax.plot([N], [F['stretch']['own']], 's', ms=4, color=c)
    ax.set_xscale('log'); ax.set_xlim(0.8, 9000); ax.set_ylim(0, 0.056); ax.set_xticks([1, 3, 10, 30, 100, 300, 1000, N]); ax.set_xticklabels(['1', '3', '10', '30', '100', '300', '1000', w('úsek', 'stretch')])
    ax.set_xlabel(w('délka kusu ℓ (po sobě jdoucí D)', 'length of the piece l (consecutive D)')); ax.set_ylabel(w('výkon k = 1 … 7', 'power at k = 1 … 7'))
    ax.set_title(w('kusy: vrstva 0 (čára), vrstvy (plně), jen losování (prázdně)', 'pieces: layer 0 (line), layers (filled), draws only (open)'), fontsize=8.5); ax.legend(loc='lower left', fontsize=7.5)
    ax = axs[1]
    for f in ('B10', 'C100', 'D100'):
        F = FA[f]; c = COL[f]; Pg = sorted(F['regression'], key=int); Pb = sorted(F['balanced'], key=int)
        ax.plot([int(p) for p in Pg], [F['regression'][p]['layer0'] for p in Pg], color=c, lw=1.2, label=f)
        ax.errorbar([int(p) for p in Pb], [F['balanced'][p]['mean'] for p in Pb], yerr=[F['balanced'][p]['se'] for p in Pb], fmt='o', ms=3.2, color=c, lw=0.8)
        ax.axhline(F['designs']['l1']['estimate'][0], color=c, lw=0.7, ls='--'); ax.axhline(F['stretch']['own'], color=c, lw=0.7, ls=':')
        for k in range(1, 8):
            pk = math.exp(k*F['l']/8)
            if 5 <= pk <= 12000: ax.plot([pk], [0.0015], marker='^', ms=4, color=c)
    ax.set_xscale('log'); ax.set_xlim(5, 12000); ax.set_ylim(0, 0.056); ax.set_xlabel(w('vyváženo na $\\chi_D(p)$ pro všechna $p \\leq P$', 'balanced on $\\chi_D(p)$ for all $p \\leq P$')); ax.set_ylabel(w('výkon k = 1 … 7', 'power at k = 1 … 7'))
    ax.set_title(w('vyvážené výběry (body), regrese bez výběrů (čára)', 'balanced samples (points), regression without samples (line)'), fontsize=8.5)
    ax.text(0.03, 0.12, w('▲ $e^{kl/8}$, k = 1 … 7 · čárkovaně náhodný výběr · tečkovaně úsek', '▲ $e^{kl/8}$, k = 1 … 7 · dashed random sample · dotted stretch'), transform=ax.transAxes, fontsize=7, color=MUT)
    ax = axs[2]; C = np.array(FA['B10']['coupling']['correlation']); Cm = C.copy(); np.fill_diagonal(Cm, np.nan); cmap = plt.get_cmap('RdBu_r').copy(); cmap.set_bad('#eeeeee'); im = ax.imshow(Cm, cmap=cmap, vmin=-1, vmax=1); ax.grid(False)
    for a in range(7):
        for b in range(7):
            if a != b: ax.text(b, a, fmt(0.0 if abs(C[a, b]) < 0.005 else C[a, b], 2, en), ha='center', va='center', fontsize=6.5, color='white' if abs(C[a, b]) > 0.45 else INK)
    ax.set_xticks(range(7)); ax.set_yticks(range(7)); ax.set_xticklabels(range(1, 8)); ax.set_yticklabels(range(1, 8)); ax.set_xlabel('k'); ax.set_ylabel("k'")
    ax.set_title(w('B10: korelace módů jedné L-funkce', 'B10: correlation of modes of one L-function'), fontsize=8.5)
    ax = axs[3]; kk = np.arange(1, 8)
    for i, f in enumerate(('B10', 'C100', 'D100')):
        W0 = IM[f]['windows'][0]; ax.bar(kk + (i - 1)*0.27, W0['gain'], width=0.26, color=COL[f], label='%s (%s %s)' % (f, w('poměr 6–7 / 1–4', 'ratio 6–7 / 1–4'), fmt(IM[f]['ratio_modes_67_to_14'], 1, en)))
        ax.plot(kk + (i - 1)*0.27, W0['null_mean'], '_', ms=7, mew=1.2, color=INK)
    ax.set_xticks(kk); ax.set_xlabel(w('mód k', 'mode k')); ax.set_ylabel(w('přírůstek vysvětleného podílu', 'increase of the explained share')); ax.set_ylim(0, 0.21); ax.legend(loc='upper left', fontsize=7.5)
    ax.set_title(w('+ $\\chi_D(n)$ 16 složených $n < e^4$ (součiny $\\chi_D(p)\\chi_D(q)$); čárka: přeházené', '+ $\\chi_D(n)$ of 16 composite $n < e^4$ (products $\\chi_D(p)\\chi_D(q)$); tick: permuted'), fontsize=8.5)
    fig.tight_layout(); save(fig, '08_layers' + sfx)
print('figure 8 written to', OUT)
