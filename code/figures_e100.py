#!/usr/bin/env python3
"""Figure 10 of study III (Section 4.8), English and Czech, PNG 300 dpi and PDF: the enlargement at |D| = 10^10 (family E100, 121 557 L-functions).
Upper left: power of the low modes kept by one consecutive stretch (exact over all starting positions) and by D mod 840, relative to random samples.
Upper right: shift of the mean first zero of even characters against USp(2N_eff) (N_eff on [0, 1/2]) over l = ln(|D|/pi), with c/l fitted without E100
and c l^-a with free a fitted with E100; odd characters grey.
Lower left: deviation of the shape, F(1) - USp(2N_eff) at x_1 = 1, for even characters, with c l^-a fitted without and with E100.
Lower right: mean first zero minus USp(2N_eff(a)) with N_eff calibrated on [0, a], E100 (thick) and C100 (thin), even (solid) and odd (dashed).
Reads reference_e100_trends.json (e100_trends.py) and reference_two_level.json from DATA_FOLDER.
usage: python3 -I figures_e100.py DATA_FOLDER OUTPUT_FOLDER"""
import sys, json, math, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
DD, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
ld = lambda n: json.load(open(os.path.join(DD, n)))
T = ld('reference_e100_trends.json'); TL = ld('reference_two_level.json')['families']
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'axes.edgecolor': '#555555',
                     'axes.linewidth': 0.8, 'xtick.color': '#333333', 'ytick.color': '#333333', 'axes.grid': True, 'grid.color': '#e3e3e3', 'grid.linewidth': 0.6,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'legend.frameon': False})
INK, MUT = '#222222', '#6b6b6b'; COL = {'B10': '#c0392b', 'C100': '#1f5fa8', 'R10': '#8e44ad', 'D100': '#2e8b57', 'E10': '#9a9a9a', 'E100': '#d35400'}
CZF = FuncFormatter(lambda v, p: ('%g' % round(v, 6)).replace('.', ',').replace('-', '−'))
def comma(fig):
    for ax in fig.axes:
        if ax.get_yscale() == 'linear' and isinstance(ax.yaxis.get_major_formatter(), ScalarFormatter): ax.yaxis.set_major_formatter(CZF)
        if ax.get_xscale() == 'linear' and isinstance(ax.xaxis.get_major_formatter(), ScalarFormatter): ax.xaxis.set_major_formatter(CZF)
def save(fig, name):
    if name.endswith('_cz'): comma(fig)
    fig.savefig(os.path.join(OUT, name + '.png'), dpi=300, bbox_inches='tight'); fig.savefig(os.path.join(OUT, name + '.pdf'), bbox_inches='tight', metadata={'CreationDate': None}); plt.close(fig)
def num(x, d, en): s = '%.*f' % (d, x); return s if en else s.replace('.', ',').replace('-', '−')
LL = np.linspace(8.5, 23.5, 200)
for en in (False, True):
    sfx = '_en' if en else '_cz'; w = lambda c, e: e if en else c
    fig, A4 = plt.subplots(2, 2, figsize=(8.6, 6.8)); a, b, c, d = A4[0, 0], A4[0, 1], A4[1, 0], A4[1, 1]
    # (a) consecutive stretch and D mod 840 relative to random samples
    C = T['conductor']; F4 = ('B10', 'C100', 'D100', 'E100'); l4 = [C[f]['l'] for f in F4]
    a.plot(l4, [C[f]['ratio_stretch_random'] for f in F4], '-', color=MUT, lw=1, zorder=1); a.plot(l4, [1 - C[f]['reduction_mod840'] for f in F4], '--', color=MUT, lw=1, zorder=1)
    for f in F4:
        a.plot(C[f]['l'], C[f]['ratio_stretch_random'], 'o', ms=7, color=COL[f], mec='white', mew=1.2, zorder=3)
        a.plot(C[f]['l'], 1 - C[f]['reduction_mod840'], 's', ms=6, color=COL[f], mec='white', mew=1.2, zorder=3)
        a.annotate(f, (C[f]['l'], C[f]['ratio_stretch_random']), textcoords='offset points', xytext=(6, -12), fontsize=8, color=COL[f])
    a.plot([], [], 'o', color=MUT, label=w('jeden úsek přes všechny začátky', 'one stretch over all starts')); a.plot([], [], 's', color=MUT, label='D mod 840')
    a.set_ylim(0, 1.08); a.axhline(1, color=INK, lw=0.6); a.set_xlabel('l = ln(|D|/π)'); a.set_ylabel(w('výkon nízkých módů / náhodný výběr', 'low-mode power / random sample'))
    a.set_title(w('souvislé úseky a zbytkové třídy', 'consecutive stretches and residue classes')); a.legend(loc='lower right')
    # (b) mean first zero of even characters
    S = T['shift']; F0 = T['shift_fit']['without_E100']; F1 = T['shift_fit']['with_E100']
    for f in ('B10', 'C100', 'R10', 'D100', 'E10', 'E100'):
        q = S[f]; b.errorbar(q['l'], q['plus']['d'], yerr=q['plus']['se'], fmt='o', ms=6 if f == 'E100' else 5, color=COL[f], mfc=COL[f] if f == 'E100' else 'white', capsize=2, lw=1, zorder=3, label=f)
        b.errorbar(q['l'] + 0.25, q['minus']['d'], yerr=q['minus']['se'], fmt='.', ms=4, color='#b5b5b5', capsize=0, lw=0.8, zorder=2)
    b.plot(LL, F0['a1']['c']/LL, '--', color=INK, lw=1, label=w('%s/l bez E100', '%s/l without E100') % num(F0['a1']['c'], 3, en))
    b.plot(LL, F1['free']['c']*LL**-F1['free']['a'], '-', color=COL['E100'], lw=1.2, label=w('c·l^−a s E100, a = %s', 'c l^-a with E100, a = %s') % num(F1['free']['a'], 2, en))
    b.axhline(0, color=INK, lw=0.6); b.set_xlabel('l = ln(|D|/π)'); b.set_ylabel(w('průměr x₁ − USp(2N_ef)', 'mean x₁ − USp(2N_eff)'))
    b.set_title(w('první nula: sudé (barevně), liché (šedě)', 'first zero: even (colour), odd (grey)')); b.legend(loc='upper right', ncol=2, fontsize=7)
    # (c) shape at x_1 = 1
    G0 = T['dF1_fit']['without_E100']; G1 = T['dF1_fit']['with_E100']; dF = T['dF1']
    for f in ('B10', 'C100', 'R10', 'D100', 'E100'):
        c.errorbar(dF[f]['l'], dF[f]['plus'][0], yerr=dF[f]['plus'][1], fmt='o', ms=6 if f == 'E100' else 5, color=COL[f], mfc=COL[f] if f == 'E100' else 'white', capsize=2, lw=1, zorder=3, label=f)
        c.errorbar(dF[f]['l'] + 0.25, dF[f]['minus'][0], yerr=dF[f]['minus'][1], fmt='.', ms=4, color='#b5b5b5', capsize=0, lw=0.8, zorder=2)
    c.plot(LL, G0['free']['c']*LL**-G0['free']['a'], '--', color=INK, lw=1, label=w('c·l^−a bez E100, a = %s', 'c l^-a without E100, a = %s') % num(G0['free']['a'], 2, en))
    c.plot(LL, G1['free']['c']*LL**-G1['free']['a'], '-', color=COL['E100'], lw=1.2, label=w('s E100, a = %s', 'with E100, a = %s') % num(G1['free']['a'], 2, en))
    c.axhline(0, color=INK, lw=0.6); c.set_xlabel('l = ln(|D|/π)'); c.set_ylabel(w('F(1) − USp(2N_ef)', 'F(1) − USp(2N_eff)'))
    c.set_title(w('tvar první nuly u x₁ = 1: sudé, liché šedě', 'shape of the first zero at x₁ = 1: even, odd grey')); c.legend(loc='lower right', ncol=2, fontsize=7)
    # (d) calibration of N_eff on [0, a]
    for f, lw_, ms_ in (('C100', 0.9, 3), ('E100', 1.6, 5)):
        for sg, ls in (('plus', '-'), ('minus', '--')):
            Q = TL[f][sg]['calibration'] if f in TL else T['calibration_E100'][sg]
            d.errorbar([q['a'] for q in Q], [q['difference'] for q in Q], yerr=[q['x1_se'] for q in Q], fmt='o' + ls, ms=ms_, lw=lw_, capsize=2, color=COL[f],
                       mfc=COL[f] if sg == 'plus' else 'white', label='%s %s' % (f, w('sudé', 'even') if sg == 'plus' else w('liché', 'odd')))
    d.axhline(0, color=INK, lw=0.6); d.axvline(0.5, color=MUT, lw=0.8, ls=':'); d.set_xlabel(w('konec intervalu kalibrace a (N_ef z hustoty na [0, a])', 'end of the calibration interval a (N_eff from the density on [0, a])'))
    d.set_ylabel(w('průměr x₁ − USp(2N_ef(a))', 'mean x₁ − USp(2N_eff(a))')); d.set_title(w('první nula podle kalibrace: E100 a C100', 'first zero by calibration: E100 and C100')); d.legend(loc='upper right', fontsize=7)
    for k, ax in zip('abcd', (a, b, c, d)): ax.text(-0.13, 1.04, '(%s)' % k, transform=ax.transAxes, fontsize=10, fontweight='bold', color=INK)
    fig.tight_layout(); save(fig, '10_e100' + sfx)
print('figure 10 written to', OUT)
