#!/usr/bin/env python3
"""
Low-lying zeros of quadratic Dirichlet L-functions, study III, Section 4.8: the enlargement at |D| = 10^10 (family E100 = E10 and the new
blocks, 121 557 L-functions) set into the trends over the conductor of Sections 4.3-4.7.

usage:  python3 -I e100_trends.py DATA_DIR OUT_DIR      (a few seconds; writes reference_e100_trends.json)
        DATA_DIR holds reference_families.json, reference_two_level.json, reference_recomputation.json, reference_explicit_weights.json and their
        counterparts for E100 (..._E100.json from compute_families.py, two_level_first_zero.py, layered_recomputation.py and explicit_weights.py
        with --families E100)

(1) the shift of the mean first zero of even characters against USp(2N_eff) (N_eff on [0, 1/2], Section 4.5) as c/l, c/l^2 and c l^-a with free a,
    for the five families of Section 4.5 (B10, C100, R10, D100, E10) and with E100 in place of E10, and the deviation of E100 from the fits without it;
(2) the same for the deviation of the shape, dF(1) = F(1) - USp(2N_eff) at x_1 = 1 (Section 4.7, two-level part);
(3) the exact ratio of one stretch over all starting positions to random samples, the reduction by D mod 840 and the ratio of the exact reduction
    by the stratifications to the line through two points (Section 4.7), and the slope of the explicit formula with and without the dual term.
Weighted least squares with the statistical errors; l = ln(|D|/pi) of the family.
"""
import sys, os, json, math, numpy as np
from scipy.stats import chi2 as CHI2
DATA, OUT_DIR = sys.argv[1], sys.argv[2]; os.makedirs(OUT_DIR, exist_ok=True)
ld = lambda n: json.load(open(os.path.join(DATA, n)))
FAM = ld('reference_families.json')['families']; FAM.update(ld('reference_families_E100.json')['families'])
TL = ld('reference_two_level.json')['families']; TL.update(ld('reference_two_level_E100.json')['families'])
RC = ld('reference_recomputation.json'); RE = ld('reference_recomputation_E100.json')
EW = ld('reference_explicit_weights.json')['families']; EW.update(ld('reference_explicit_weights_E100.json')['families'])
def l_of(f): return FAM[f]['plus']['mean_ln_D'] - math.log(math.pi)
def fit_power(l, y, s, a):
    x = l**-a; c = float(np.sum(x*y/s**2)/np.sum(x*x/s**2)); return c, float(1/math.sqrt(np.sum(x*x/s**2))), float(np.sum(((y - c*x)/s)**2))
def fits(rows):
    l = np.array([r[1] for r in rows]); y = np.array([r[2] for r in rows]); s = np.array([r[3] for r in rows]); out = {}
    for a in (1, 2):
        c, ce, ch = fit_power(l, y, s, a); out['a%d' % a] = dict(c=c, se=ce, chi2=ch, dof=len(rows) - 1, p=float(CHI2.sf(ch, len(rows) - 1)))
    grid = np.round(np.arange(0.2, 6.0001, 0.005), 3); chs = np.array([fit_power(l, y, s, a)[2] for a in grid]); i = int(np.argmin(chs)); ok = grid[chs <= chs[i] + 1]
    c, ce, ch = fit_power(l, y, s, grid[i])
    out['free'] = dict(a=float(grid[i]), a_lo=float(ok.min()), a_hi=float(ok.max()), c=c, chi2=ch, dof=len(rows) - 2, p=float(CHI2.sf(ch, max(len(rows) - 2, 1))))
    return out
def pred(F, l): return {k: v['c']*l**-(1 if k == 'a1' else 2 if k == 'a2' else v['a']) for k, v in F.items()}
OUT = dict(study='III', section='4.8', families_shift_45=['B10', 'C100', 'R10', 'D100', 'E10'], families_shift_48=['B10', 'C100', 'R10', 'D100', 'E100'])
# ---- (1) mean first zero, even and odd, N_eff on [0, 1/2]
sh = {}
for f in ('B10', 'C10', 'C100', 'R10', 'D10', 'D100', 'E10', 'E100'):
    sh[f] = dict(l=l_of(f), **{sg: dict(d=FAM[f][sg]['x1_mean'] - FAM[f][sg]['x1_usp2n_eff'], se=FAM[f][sg]['x1_se'], ks=FAM[f][sg]['ks_usp2n_eff'], ks_critical=FAM[f][sg]['ks_critical'], n=FAM[f][sg]['n'])
                                for sg in ('plus', 'minus')})
OUT['shift'] = sh
R0 = [(f, sh[f]['l'], sh[f]['plus']['d'], sh[f]['plus']['se']) for f in OUT['families_shift_45']]; R1 = [(f, sh[f]['l'], sh[f]['plus']['d'], sh[f]['plus']['se']) for f in OUT['families_shift_48']]
F0 = fits(R0); F1 = fits(R1); lE = sh['E100']['l']; P0 = pred(F0, lE)
OUT['shift_fit'] = dict(without_E100=F0, with_E100=F1, E100=dict(value=sh['E100']['plus']['d'], se=sh['E100']['plus']['se'], predicted_without_E100=P0,
                        z=({k: (sh['E100']['plus']['d'] - v)/sh['E100']['plus']['se'] for k, v in P0.items()})))
# ---- (2) shape: dF(1) of even characters (two-level part, Section 4.7)
dF = {f: dict(l=l_of(f), plus=[TL[f]['plus']['rows']['1.0']['dF'], TL[f]['plus']['rows']['1.0']['F'][1]], minus=[TL[f]['minus']['rows']['1.0']['dF'], TL[f]['minus']['rows']['1.0']['F'][1]],
              plus_05=[TL[f]['plus']['rows']['0.5']['dF'], TL[f]['plus']['rows']['0.5']['F'][1]], plus_07=[TL[f]['plus']['rows']['0.7']['dF'], TL[f]['plus']['rows']['0.7']['F'][1]]) for f in TL}
OUT['dF1'] = dF
S0 = [(f, dF[f]['l'], dF[f]['plus'][0], dF[f]['plus'][1]) for f in ('B10', 'C100', 'R10', 'D100')]; S1 = S0 + [('E100', dF['E100']['l'], dF['E100']['plus'][0], dF['E100']['plus'][1])]
G0 = fits(S0); G1 = fits(S1); Q0 = pred(G0, lE)
OUT['dF1_fit'] = dict(families_without=['B10', 'C100', 'R10', 'D100'], without_E100=G0, with_E100=G1, E100=dict(value=dF['E100']['plus'][0], se=dF['E100']['plus'][1], predicted_without_E100=Q0,
                      z=({k: (dF['E100']['plus'][0] - v)/dF['E100']['plus'][1] for k, v in Q0.items()})))
OUT['calibration_E100'] = TL['E100']['plus']['calibration'] and dict(plus=TL['E100']['plus']['calibration'], minus=TL['E100']['minus']['calibration'], even_minus_odd=TL['E100']['even_minus_odd_by_calibration'])
# ---- (3) consecutive stretches, stratifications, line, explicit formula
cons = {}
for f, J in (('B10', RC), ('C100', RC), ('D100', RC), ('E100', RE)):
    F = J['families'][f]; D = F['designs']; r = D['random']['estimate']['low']; m8 = D['mod840']['estimate']['low']
    cons[f] = dict(l=F['l'], n=F['n'], random=r, mod840=m8, stretch_all_starts=F['stretch']['own'][0], ratio_stretch_random=J['z_44'][f]['stretch_all_starts_ratio'],
                   reduction_mod840=(r[0] - m8[0])/r[0], z_mod840=J['z_44'][f]['mod840'], n_below_3=int(sum(1 for k, v in J['z_44'][f].items() if k.startswith('mod') and v < -3)),
                   layer0_reduction_ratio_to_line=J['line_layer0_exact'][f]['mean_ratio_of_reductions'], W100_all_starts=F['W100_all_starts'],
                   slope_diagonal=EW[f]['regression']['slope_chi'], slope_dual=EW[f]['regression']['slope_chi_with_dual'])
OUT['conductor'] = cons; OUT['line_E100'] = RE['line']['layered_all_starts']
# ---- E10 against E100
OUT['E10_E100'] = {f: {k: FAM[f]['all'][k] for k in ('n', 'chi2_005_ratios', 'chi2_005_without_AD', 'chi2_005_usp2n', 'chi2_005_usp', 'density_0_05', 'density_0_05_se', 'density_0_05_ratios', 'density_0_05_usp',
                                                       'x1_mean', 'x1_se', 'x1_usp2n_eff', 'N_eff')} for f in ('E10', 'E100')}
for f in ('E10', 'E100'):
    for sg in ('plus', 'minus'): OUT['E10_E100'][f][sg] = {k: FAM[f][sg][k] for k in ('n', 'chi2_005_ratios', 'x1_mean', 'x1_se', 'x1_usp2n_eff', 'N_eff', 'ks_usp2n_eff', 'ks_critical')}
json.dump(OUT, open(os.path.join(OUT_DIR, 'reference_e100_trends.json'), 'w'), indent=1)
print('shift of even characters: c/l without E100 c = %.4f +- %.4f (p %.2f), with E100 %.4f (p %.2f); free power %.2f (%.2f-%.2f) -> %.2f (%.2f-%.2f); E100 %+.4f +- %.4f, z against c/l %.1f' % (
      F0['a1']['c'], F0['a1']['se'], F0['a1']['p'], F1['a1']['c'], F1['a1']['p'], F0['free']['a'], F0['free']['a_lo'], F0['free']['a_hi'], F1['free']['a'], F1['free']['a_lo'], F1['free']['a_hi'],
      sh['E100']['plus']['d'], sh['E100']['plus']['se'], OUT['shift_fit']['E100']['z']['a1']))
print('dF(1) of even characters: free power without E100 %.2f (%.2f-%.2f), predicted for E100 %.4f, measured %.4f +- %.4f (z %.1f); with E100 free %.2f (p %.2f), 1/l p %.3f, 1/l^2 p %.3f' % (
      G0['free']['a'], G0['free']['a_lo'], G0['free']['a_hi'], Q0['free'], dF['E100']['plus'][0], dF['E100']['plus'][1], OUT['dF1_fit']['E100']['z']['free'], G1['free']['a'], G1['free']['p'], G1['a1']['p'], G1['a2']['p']))
for f, c in cons.items():
    print('%-5s l %.2f stretch/random %.3f, D mod 840 %.1f %% (z %.1f), %d moduli below -3, stratification/line %.2f, slope diagonal %.3f dual %s' % (
          f, c['l'], c['ratio_stretch_random'], 100*c['reduction_mod840'], c['z_mod840'], c['n_below_3'], c['layer0_reduction_ratio_to_line'], c['slope_diagonal'][0], c['slope_dual'] and '%.4f' % c['slope_dual'][0]))
