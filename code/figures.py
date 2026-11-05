#!/usr/bin/env python3
"""Figures of study III (light background, English and Czech), PNG 300 dpi and PDF, from the outputs of compute_families.py (and, for the right panel of
Figure 5, reference_pieces_repeat.json of pieces_repeat.py when it lies next to reference_pieces.json).
usage: python3 -I figures.py reference_families.json reference_subsets.json reference_pieces.json OUTPUT_FOLDER"""
import sys, json, math, os, numpy as np
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter
RF, RS, RP, OUT = sys.argv[1:5]; os.makedirs(OUT, exist_ok=True)
F = json.load(open(RF)); S = json.load(open(RS)); P = json.load(open(RP)); FA = F['families']
RP2 = os.path.join(os.path.dirname(os.path.abspath(RP)), 'reference_pieces_repeat.json'); PR2 = json.load(open(RP2)) if os.path.exists(RP2) else None   # test with pieces with the error of every point (pieces_repeat.py)
RX = os.path.join(os.path.dirname(os.path.abspath(RP)), 'reference_first_zero_crossing.json'); PX = json.load(open(RX)) if os.path.exists(RX) else None   # crossing of the first zero (first_zero_crossing.py)
plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 9, 'axes.titlesize': 10, 'axes.labelsize': 9, 'legend.fontsize': 8, 'axes.edgecolor': '#555555',
                     'axes.linewidth': 0.8, 'xtick.color': '#333333', 'ytick.color': '#333333', 'axes.grid': True, 'grid.color': '#e3e3e3', 'grid.linewidth': 0.6,
                     'figure.facecolor': 'white', 'axes.facecolor': 'white', 'savefig.facecolor': 'white', 'legend.frameon': False})
INK, MUT, RAT, USP, NOA, CON, RND, MOD, EVN, ODD = '#222222', '#6b6b6b', '#1f5fa8', '#d07a1a', '#b2589a', '#c0392b', '#3b6fb6', '#7b5ea7', '#c0392b', '#1f5fa8'
RFAM = '#7fb3e6'   # random families R, R10 in figures 2 and 3 (light blue)
OFF2 = {'C100': (-26, -11), 'R10': (5, 2), 'E10': (5, -9)}                                                   # label offsets in figure 2 (points close together)
OFF3 = {'C10': (-20, 5), 'C100': (-12, -13), 'R': (5, 3), 'R10': (5, -10), 'D10': (-24, 5), 'D100': (5, -10), 'E': (-14, 6), 'E10': (5, -10)}   # figure 3
OFF7 = {'C100': (-8, -13), 'R10': (5, 4)}                                                                    # figure 7
ORDER = [f for f in ['A', 'B', 'B10', 'C10', 'C100', 'D10', 'D100', 'E', 'E10', 'R', 'R10'] if f in FA]
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
XB = F['xb']; NB = F['nb']; XC = (np.arange(NB) + 0.5)*XB; USPB = np.array(F['usp'])
def dens(f, s):
    q = FA[f][s]; H = np.array(q['hist_0025']).reshape(NB, 2).sum(1); n = q['n']; return H/(n*XB), np.sqrt(np.maximum(H, 1))/(n*XB), np.array(q['prediction_0025']).reshape(NB, 2).mean(1)
for en in (False, True):
    sfx = '_en' if en else '_cz'; w = lambda c, e: e if en else c
    # 1) density of the low zeros: C100 and E10 against the ratios conjecture and USp
    fig, axs = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for ax, f, tit in ((axs[0], 'C100', w('C100: |D| = 10⁶ … 2·10⁶, 607 941 L', 'C100: |D| = 10⁶ … 2·10⁶, 607,941 L')), (axs[1], 'E10', w('E10: |D| = 10¹⁰ … 10¹⁰ + 2·10⁴, 12 153 L', 'E10: |D| = 10¹⁰ … 10¹⁰ + 2·10⁴, 12,153 L'))):
        d, e, p = dens(f, 'all'); ax.plot(XC, USPB, color=USP, lw=1.2, label=w('limita USp (Katz–Sarnak)', 'USp limit (Katz–Sarnak)')); ax.plot(XC, p, color=RAT, lw=1.5, label=w('domněnka poměrů', 'ratios conjecture'))
        ax.errorbar(XC, d, yerr=e, fmt='o', ms=2.2, color=INK, lw=0.6, label=w('nuly (PARI/GP)', 'zeros (PARI/GP)')); ax.set_xlabel(w('rozvinutá výška $x = \\theta_D(\\gamma)/\\pi$', 'unfolded height $x = \\theta_D(\\gamma)/\\pi$')); ax.set_title(tit, loc='left')
        q = FA[f]['all']; ax.text(0.97, 0.05, 'χ²/80 = %s · USp %s' % (fmt(q['k05']['chi2_ratios'], 2, en), fmt(q['chi2_005_usp'], 0 if q['chi2_005_usp'] > 10 else 1, en)), transform=ax.transAxes, ha='right', fontsize=8, color=MUT)
    axs[0].set_ylabel(w('hustota nul (koše 0,05)', 'density of zeros (bins 0.05)')); axs[0].legend(loc='lower right', bbox_to_anchor=(1, 0.1)); fig.tight_layout(); save(fig, '01_density' + sfx)
    # 2) chi2 per bin of all families against ln|D|: ratios conjecture with the bootstrap band, null models (log)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4))
    for f in ORDER:
        q = FA[f]['all']; x = q['mean_ln_D']; B = q['bootstrap']['k05']; c = q['k05']['chi2_ratios']
        a1.plot([x, x], [B['q05'], B['q95']], color='#bbbbbb', lw=4, solid_capstyle='butt'); a1.plot(x, c, 'o', color=RFAM if f.startswith('R') else RAT, ms=4.5)
        a1.annotate(f, (x, c), textcoords='offset points', xytext=OFF2.get(f, (4, 3)), fontsize=7, color=MUT)
        a2.plot(x, q['chi2_005_usp'], 's', color=USP, ms=4); a2.plot(x, q['chi2_005_without_AD'], '^', color=NOA, ms=4); a2.plot(x, c, 'o', color=RAT, ms=4)
    a1.axhline(0.95, color=MUT, lw=0.8, ls='--'); a1.set_ylim(0.3, 1.5); a1.set_xlabel('ln |D|'); a1.set_ylabel(w('χ² na koš (0,05), domněnka poměrů', 'χ² per bin (0.05), ratios conjecture'))
    a1.set_title(w('šedě 5–95 % bootstrapu, čárkovaně nezávislé L', 'grey 5–95 % of the bootstrap, dashed independent L'), loc='left', fontsize=8.5)
    a2.set_yscale('log'); a2.axhline(0.95, color=MUT, lw=0.8, ls='--'); a2.set_xlabel('ln |D|'); a2.set_ylabel(w('χ² na koš (0,05)', 'χ² per bin (0.05)'))
    a2.plot([], [], 's', color=USP, label=w('limita USp', 'USp limit')); a2.plot([], [], '^', color=NOA, label=w('bez $A_D$', 'without $A_D$')); a2.plot([], [], 'o', color=RAT, label=w('domněnka poměrů', 'ratios conjecture')); a2.legend(loc='upper right')
    fig.tight_layout(); save(fig, '02_chi2_families' + sfx)
    # 3) approach to USp: density in [0, 1/2] against ln|D| with the curve of the conjecture for one D
    fig, ax = plt.subplots(figsize=(6.2, 3.5)); K = np.array(F['curve_one_D'])
    ax.plot(K[:, 0], 0.5*(K[:, 1] + K[:, 2]), color=RAT, lw=1.6, label=w('domněnka poměrů pro jedno D (průměr znamének)', 'ratios conjecture for one D (mean of the signs)'))
    ax.plot(K[:, 0], K[:, 1], color=RAT, lw=0.8, ls='--', label='D > 0'); ax.plot(K[:, 0], K[:, 2], color=RAT, lw=0.8, ls=':', label='D < 0'); ax.axhline(FA['C100']['all']['density_0_05_usp'], color=USP, lw=1.2, label=w('limita USp', 'USp limit'))
    for f in ORDER:
        q = FA[f]['all']; ax.errorbar(q['mean_ln_D'], q['density_0_05'], yerr=q['density_0_05_se'], fmt='o', ms=4, color=RFAM if f.startswith('R') else INK, lw=0.8); ax.annotate(f, (q['mean_ln_D'], q['density_0_05']), textcoords='offset points', xytext=OFF3.get(f, (5, 4)), fontsize=7, color=MUT)
    ax.set_xlabel('ln |D|'); ax.set_ylabel(w('hustota nul v [0; 0,5]', 'density of zeros in [0, 0.5]')); ax.legend(loc='upper right'); fig.tight_layout(); save(fig, '03_approach' + sfx)
    # 4) chi2 of subsets of 6 074 L: consecutive bands, random samples, samples with the classes D mod 840
    fig, ax = plt.subplots(figsize=(7.0, 3.5)); rng = np.random.default_rng(1); LIM = {'B10': 81, 'C100': 100, 'D100': 30}
    for i, f in enumerate(['B10', 'C100', 'D100']):
        sub = S['families'][f]['subsets']
        for j, (typ, c, lab) in enumerate((('consecutive', CON, w('souvislá pásma', 'consecutive bands')), ('random', RND, w('náhodné výběry', 'random samples')), ('mod840', MOD, w('třídy D mod 840', 'classes D mod 840')))):
            v = np.array([q['chi2_005'] for q in sub if q['type'] == typ])[: (LIM[f] if typ != 'consecutive' else None)]; x0 = i + (j - 1)*0.27
            ax.plot(x0 + rng.uniform(-0.09, 0.09, len(v)), v, 'o', ms=2.2, color=c, alpha=0.75, label=lab if i == 0 else None)
            ax.errorbar(x0, v.mean(), yerr=2*v.std(ddof=1)/math.sqrt(len(v)), fmt='_', ms=16, mew=2, color=INK, lw=1.2, capsize=0)
        if f == 'B10':
            b = [q for q in sub if q['type'] == 'consecutive'][0]['chi2_005']; ax.plot(i - 0.27, b, 'o', ms=9, mfc='none', mec=CON, mew=1.4); ax.annotate(w('rodina B', 'family B') + ' ' + fmt(b, 2, en), (i - 0.27, b), textcoords='offset points', xytext=(8, -3), fontsize=8, color=CON)
    ax.axhline(0.95, color=MUT, lw=0.8, ls='--'); ax.set_xticks([0, 1, 2]); ax.set_xticklabels(['B10 (|D| 10⁴ … 10⁵)', 'C100 (10⁶ … 2·10⁶)', 'D100 (10⁸ … 10⁸ + 10⁵)'])
    ax.set_ylabel(w('χ² na koš (0,05)', 'χ² per bin (0.05)')); ax.set_ylim(0.3, 1.55); ax.legend(loc='upper right', ncol=3); ax.set_title(w('Podmnožiny po 6 074 L; čárky průměr ± 2 chyby', 'Subsets of 6,074 L; ticks mean ± 2 errors'), loc='left')
    fig.tight_layout(); save(fig, '04_subsets' + sfx)
    # 5) spectrum of the deviations (C100) and the test with pieces
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4), gridspec_kw={'width_ratios': [1.25, 1]}); sub = S['families']['C100']['subsets']
    def sm(a): return np.array([np.mean(a[max(1, k - 1):min(len(a), k + 2)]) for k in range(len(a))])
    for typ, c, lab in (('random', RND, w('náhodné výběry', 'random samples')), ('mod840', MOD, w('třídy D mod 840', 'classes D mod 840')), ('consecutive', CON, w('souvislá pásma', 'consecutive bands'))):
        A_ = np.array([q['power'] for q in sub if q['type'] == typ][:100]).mean(0); k = np.arange(1, 41); a1.plot(k, sm(A_)[1:41], color=c, lw=1.6 if typ == 'consecutive' else 1.1, ls='--' if typ == 'mod840' else '-', label=lab)
    l = FA['C100']['all']['mean_ln_D'] - math.log(math.pi); band = [q for q in sub if q['type'] == 'consecutive'][0]; uc = math.log(band['D_abs'][1] - band['D_abs'][0])/l
    for u, ls in ((uc, '--'), (2*uc, ':')): a1.axvline(8*u, color=CON, lw=0.9, ls=ls)
    a1.text(8*uc + 0.4, a1.get_ylim()[1]*0.9 if a1.get_ylim()[1] else 0.012, '$u_c$', color=CON, fontsize=8); a1.set_xlabel(w('index kosinové transformace k (frekvence u = k/8)', 'index of the cosine transform k (frequency u = k/8)')); a1.set_ylabel(w('průměrný výkon', 'mean power'))
    a1.set_title(w('C100: odchylky (data − domněnka)/chyba, 80 košů', 'C100: deviations (data − conjecture)/error, 80 bins'), loc='left', fontsize=8.5); a1.legend(loc='lower right')
    for f, c in (('B10', CON), ('C100', RND)):
        R = P['families'][f]; ms = sorted(int(m) for m in R['m']); low = [sum(R['m'][str(m)]['bands'][1:3]) for m in ms]
        if PR2 is None: a2.plot(ms, low, 'o-', color=c, ms=3.5, lw=1.2, label=f); continue
        a2.plot([m*1.12 for m in ms], low, 'o', mfc='white', mec=c, mew=0.9, ms=3.6, alpha=0.75, zorder=2)                     # first run (60 / 150 subsets), shifted to the right
        Q = PR2['repetition']['families'][f]['m']; m2 = sorted(int(m) for m in Q); y2 = [Q[str(m)]['power_k1_7'] for m in m2]; e2 = [Q[str(m)]['power_k1_7_se'] for m in m2]
        a2.errorbar(m2, y2, yerr=e2, fmt='o-', color=c, ms=3.5, lw=1.2, elinewidth=0.9, capsize=2, zorder=3, label=f)
    if PR2 is not None:
        n1 = PR2['first_run']['subsets']; a2.plot([], [], 'o', mfc='white', mec=MUT, mew=0.9, ms=3.6, label=w('první běh: %d / %d' % (n1['B10'], n1['C100']), 'first run: %d / %d' % (n1['B10'], n1['C100'])))
    a2.set_xscale('log'); a2.set_xlabel(w('počet souvislých kusů m', 'number of consecutive pieces m')); a2.set_ylabel(w('výkon na k = 1 … 7', 'power at k = 1 … 7')); hl_ = a2.get_legend_handles_labels(); ix_ = sorted(range(len(hl_[1])), key=lambda i: hl_[1][i].startswith(('první', 'first'))); a2.legend([hl_[0][i] for i in ix_], [hl_[1][i] for i in ix_], loc='lower right')
    a2.set_title(w('6 074 L z m kusů (m = 6 074: náhodný výběr)', '6,074 L from m pieces (m = 6,074: random)') + ('' if PR2 is None else w('\n%d výběrů na bod, ±1 chyba' % PR2['repetition']['subsets'], '\n%d samples per point, ±1 error' % PR2['repetition']['subsets'])), loc='left', fontsize=8.5); fig.tight_layout(); save(fig, '05_spectrum_pieces' + sfx)
    # 6) first zero: C100 distribution against USp(2N_ef) for both signs; deviation of the mean against l
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(7.4, 3.4)); SG = np.array(F['sg'])
    for s, c, lab in (('plus', EVN, w('sudé (D > 0)', 'even (D > 0)')), ('minus', ODD, w('liché (D < 0)', 'odd (D < 0)'))):
        q = FA['C100'][s]; a1.plot(SG, np.array(q['x1_cdf']) - np.array(q['x1_cdf_usp2n_eff']), color=c, lw=1.4, label=lab + ' · KS ' + fmt(q['ks_usp2n_eff'], 4, en))
    a1.axhspan(-FA['C100']['plus']['ks_critical'], FA['C100']['plus']['ks_critical'], color='#2a9d6f', alpha=0.12, lw=0); a1.axhline(0, color=MUT, lw=0.8); a1.set_xlim(0, 2); a1.set_ylim(-0.0215, 0.0048)
    if PX is not None:
        qx = PX['families']['C100']['even']; a1.axvline(0.5, color=MUT, lw=0.8, ls='--')
        a1.errorbar([qx['crossing']], [0], xerr=[qx['crossing_se']], fmt='o', ms=4, mfc='white', mec='#6a3d9a', ecolor='#6a3d9a', mew=1.1, lw=1.1, capsize=2.5, zorder=5,
                    label=w('průsečík ', 'crossing ') + fmt(qx['crossing'], 3, en) + ' ± ' + fmt(qx['crossing_se'], 3, en))
    a1.set_xlabel(w('první nula x₁', 'first zero x₁')); a1.set_ylabel(w('F(x₁) data − USp(2$N_{\\rm ef}$)', 'F(x₁) data − USp(2$N_{\\rm eff}$)')); a1.legend(loc='lower right'); a1.set_title(w('C100; zeleně kritická hodnota KS', 'C100; green the KS critical value') + ('' if PX is None else w('; čárkovaně x₁ = 1/2', '; dashed x₁ = 1/2')), loc='left', fontsize=8.5)
    nez = ['A', 'B10', 'C100', 'D100', 'E10', 'R10']; ll = np.linspace(6, 23, 100)
    for s, c, dx in (('plus', EVN, -0.15), ('minus', ODD, 0.15)):
        L_ = np.array([2*FA[f][s]['N_half_l'] for f in nez]); D_ = np.array([FA[f][s]['x1_mean'] - FA[f][s]['x1_usp2n_eff'] for f in nez]); E_ = np.array([FA[f][s]['x1_se'] for f in nez])
        a2.errorbar(L_ + dx, D_, yerr=E_, fmt='o', ms=4, color=c, lw=0.9); cc = np.sum(D_/L_/E_**2)/np.sum(1/L_**2/E_**2)
        a2.plot(ll, cc/ll, color=c, lw=1.0, ls='-' if s == 'plus' else ':', label=(w('sudé: c/l, c = ', 'even: c/l, c = ') if s == 'plus' else w('liché: c/l, c = ', 'odd: c/l, c = ')) + fmt(cc, 3, en).replace('-', '−'))
        if s == 'plus':
            for f, x, y in zip(nez, L_, D_): a2.annotate(f, (x + dx, y), textcoords='offset points', xytext=OFF7.get(f, (4, 4)), fontsize=7, color=MUT)
    a2.axhline(0, color=MUT, lw=0.8); a2.set_xlabel('l = ln(|D|/π)'); a2.set_ylabel(w('x₁ data − USp(2$N_{\\rm ef}$)', 'x₁ data − USp(2$N_{\\rm eff}$)')); a2.legend(loc='upper right'); fig.tight_layout(); save(fig, '07_first_zero' + sfx)
print('figures written to', OUT)
