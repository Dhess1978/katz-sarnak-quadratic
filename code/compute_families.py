#!/usr/bin/env python3
"""
Low-lying zeros of quadratic Dirichlet L-functions: ratios conjecture, consecutive families and the first zero (study III).
Computes every number of the article from the zeros produced by ks_nuly.py (PARI/GP), read with read_zeros.py.

usage:  python3 -I compute_families.py ZEROS_ROOT OUT_DIR
        python3 -I compute_families.py ZEROS_ROOT OUT_DIR --families E100      (Section 4.8: stage 1 only, writes reference_families_E100.json, about 2 minutes)
        ZEROS_ROOT: folder with the sets written by ks_nuly.py (e.g. velky_..., dohoda_..., male_..., kalibrace_... side by side)
        OUT_DIR:    receives reference_families.json, reference_subsets.json, reference_pieces.json   (about 5 minutes, 7 GB RAM is enough)

Stage 1 (families, seed 3463): for each family and sign of D the histogram of the unfolded zeros x = theta_D(gamma)/pi in bins of 0.025 up to x = 4,
  the one-level density of the ratios conjecture (Conrey-Snaith 2007, Theorem 3.1; for D < 0 with Gamma(3/4 + it/2), our analogue) averaged over all D of the
  family (linear interpolation in l = ln(|D|/pi) between nodes 0.002 apart), the null models (without the arithmetic factor A_D, USp(2N) with N = l/2, the USp limit),
  chi^2 per bin with Poisson errors at bin widths 0.025 / 0.05 / 0.1, the chi^2 that independent L-functions would give (from the scatter between them),
  a bootstrap over L-functions (200 resamplings), a test without bins (cumulative count), and the first zero against the USp limit, USp(2N) and USp(2N_ef),
  N_ef chosen so that the density of USp(2N_ef) in [0, 1/2] equals that of the ratios conjecture (from the theory, not from the data).
Stage 2 (subsets, seed 34630): subsets of 6 074 L (the size of family B) in B10, C100, D100 - consecutive bands (sorted by |D|), random samples, and random samples
  with the same composition of classes (sign, D mod M), M = 8, 24, 120, 840, as the whole family; chi^2 per bin (0.05 and 0.1) and the cosine transform of the
  80 normalised deviations (data - prediction)/error.
Stage 3 (pieces, seed 346301): subsets of 6 074 L made of m consecutive pieces at random positions, m = 1 ... 6074; mean power of the cosine transform.
"""
import sys, os, json, math, time, numpy as np, mpmath as mp, scipy.special as sp, scipy.fft as FFT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
t0 = time.time(); ARGS = [a for a in sys.argv[1:] if not a.startswith('--')]; ROOT, OUT_DIR = ARGS[0], ARGS[1]; os.makedirs(OUT_DIR, exist_ok=True)
ONLY = sys.argv[sys.argv.index('--families') + 1].split(',') if '--families' in sys.argv else None; SUFFIX = ('_' + '_'.join(ONLY)) if ONLY else ''
if ONLY: ARGS = ARGS[:2]
DT = 0.002; TG = np.round(np.arange(0, 8.0 + 1e-9, DT), 4); NT = len(TG); XF = 0.025; NF = 160; XB = 0.05; NB = 80; NL = 24; BOOT = 200; LB = 0.002; NPAS = 6074
ORDER_ALL = ['A', 'B', 'B10', 'C10', 'C100', 'D10', 'D100', 'E', 'E10', 'R', 'R10'] if not ONLY else ONLY   # E100 of the study = E10 + E100 of the generator (Section 4.8)
FAM = {}; SRC = {}; RF = RZ.families(ROOT)
for f in ORDER_ALL:
    if f not in RF: continue
    Ds, G = [], []
    for D, g in RZ.study_zeros(f, ROOT): Ds.append(D); G.append(g)
    FAM[f] = (np.array(Ds, dtype=np.int64), G); SRC[f] = '+'.join(RF[q][0] for q in RZ.STUDY.get(f, (f,)))
ORDER = [f for f in ORDER_ALL if f in FAM]
print('families', {f: len(FAM[f][0]) for f in ORDER}, round(time.time() - t0), 's', flush=True)
# ---------------------------------------------------------------- the ratios conjecture on a grid in t (mpmath 30 digits, primes up to 2e5)
Mp = 200000; s = np.ones(Mp + 1, bool); s[:2] = False
for p in range(2, int(Mp**0.5) + 1):
    if s[p]: s[p*p::p] = False
P = np.nonzero(s)[0].astype(float); lp = np.log(P)
A1 = np.empty(NT, complex); A2 = np.empty(NT, complex)
for i, tt in enumerate(TG):
    z = np.exp(2j*tt*lp); A1[i] = np.exp(np.sum(np.log(1 - (z - 1)/(P*P - 1)))); A2[i] = np.sum(lp/((P + 1)*(P*z - 1)))
mp.mp.dps = 30; ZL = np.zeros(NT, complex); ZR = np.zeros(NT, complex)
for i, tt in enumerate(TG):
    if i == 0: continue
    s1 = mp.mpc(1, 2*float(tt)); ZL[i] = complex(mp.zeta(s1, derivative=1)/mp.zeta(s1)); ZR[i] = complex(mp.zeta(mp.mpc(1, -2*float(tt))))
GR = {k: np.exp(sp.loggamma(0.25 + 0.5*k - 0.5j*TG) - sp.loggamma(0.25 + 0.5*k + 0.5j*TG)) for k in (0, 1)}
IMLG = {k: np.imag(sp.loggamma(0.25 + 0.5*k + 0.5j*TG)) for k in (0, 1)}
Z1 = ZL + A2; Z2 = {k: GR[k]*ZR*A1 for k in (0, 1)}; Z2_0 = {k: GR[k]*ZR for k in (0, 1)}
def theta(d, t): k = 0 if d > 0 else 1; return 0.5*np.asarray(t)*math.log(abs(d)/math.pi) + np.imag(sp.loggamma(0.25 + 0.5*k + 0.5j*np.asarray(t)))
def predl(l, k, ar=True, xb=XF, nb=NF):
    """bin averages of the one-level density of the ratios conjecture for one D (l = ln(|D|/pi), k = 0 for D > 0, 1 for D < 0); ar=False drops A_D"""
    a, b = (Z1, Z2[k]) if ar else (ZL, Z2_0[k]); c = np.real(a - np.exp(-1j*TG*l)*b)/math.pi; c[0] = 3*c[1] - 3*c[2] + c[3]
    C = np.concatenate([[0], np.cumsum(0.5*(c[1:] + c[:-1])*DT)])
    th = (0.5*TG*l + IMLG[k])/math.pi; te = np.interp(np.arange(nb + 1)*xb, th, TG)
    return 1 + np.diff(np.interp(te, TG, C))/xb
def W_N(x, N): y = np.pi*x/N; return np.where(x < N, 1 + 1/(2*N) - np.sin((2*N + 1)*y)/(2*N*np.sin(np.minimum(y, np.pi - 1e-12))), np.nan)
xs = np.linspace(0, NB*XB, NB*200 + 1); xs[0] = 1e-9
def binavg(f): v = f(xs); seg = 0.5*(v[1:] + v[:-1])*np.diff(xs); return seg.reshape(NB, 200).sum(1)/XB
B_usp = binavg(lambda x: 1 - np.sinc(2*x))
xs2 = np.linspace(0, NF*XF, NF*100 + 1); xs2[0] = 1e-9; vv = 1 - np.sinc(2*xs2); B_usp160 = np.array([np.trapezoid(vv[100*b:100*(b + 1) + 1], xs2[100*b:100*(b + 1) + 1])/XF for b in range(NF)])
agg = lambda a, m: a.reshape(-1, m).sum(1)
aggm = lambda a, m: a.reshape(-1, m).mean(1)
CACHE = {}
FUN = {'pr': lambda l, k: predl(l, k), 'pr0': lambda l, k: predl(l, k, False), 'w80': lambda l, k: binavg(lambda xx: W_N(xx, 0.5*l))}
def node(fn, k, j):
    key = (fn, k, j)
    if key not in CACHE: CACHE[key] = FUN[fn](j*LB, k)
    return CACHE[key]
def mean_over(fn, L, S):
    """mean of FUN[fn](l_D, k_D) over the discriminants (nodes in l every LB, linear interpolation; error of second order)"""
    tot = None
    for k in (0, 1):
        m = S == k
        if not m.any(): continue
        li = L[m]; j = np.floor(li/LB).astype(np.int64); u = li/LB - j
        uj, inv = np.unique(j, return_inverse=True); cnt = np.bincount(inv).astype(float); su = np.bincount(inv, weights=u)
        for q, jj in enumerate(uj):
            v = (cnt[q] - su[q])*node(fn, k, int(jj)) + su[q]*node(fn, k, int(jj) + 1); tot = v if tot is None else tot + v
    return tot/len(L)
# ---------------------------------------------------------------- Fredholm determinants for the first zero (Gauss-Legendre 40; USp(2N): kernel S_N(x - y) - S_N(x + y))
XG, WG = np.polynomial.legendre.leggauss(40)
def E0(s_, N=None):
    if s_ <= 0: return 1.0
    x = 0.5*s_*(XG + 1); w = 0.5*s_*WG
    if N is None: S_ = lambda z: np.sinc(z)
    else: S_ = lambda z: np.where(np.abs(z) < 1e-12, (2*N + 1)/(2*N), np.sin((2*N + 1)*np.pi*z/(2*N))/(2*N*np.sin(np.pi*z/(2*N)) + 1e-300))
    K = S_(x[:, None] - x[None, :]) - S_(x[:, None] + x[None, :]); A = np.eye(len(x)) - np.sqrt(w)[:, None]*K*np.sqrt(w)[None, :]
    return float(np.linalg.det(A))
SG = np.round(np.arange(0, 3.0001, 0.02), 4)
def first_zero_model(N=None):
    E = np.array([E0(s_, N) if (N is None or s_ < 0.98*N) else 0.0 for s_ in SG]); return 1 - E, float(np.trapezoid(E, SG))
F_usp, m_usp = first_zero_model()
def d0_2n(N): v = binavg(lambda xx: W_N(xx, N)); return float(np.mean(v[:10]))
def n_eff(d0p):
    lo, hi = 0.55, 80.0
    for _ in range(60):
        mid = 0.5*(lo + hi)
        if d0_2n(mid) > d0p: lo = mid
        else: hi = mid
    return 0.5*(lo + hi)
print('grid ready; USp mean first zero %.5f' % m_usp, round(time.time() - t0), 's', flush=True)
def chi05(H80, n, pm80):
    dens = H80/(n*XB); se = np.sqrt(np.maximum(H80, 1))/(n*XB); return float(np.mean(((dens - pm80)/se)**2))
def indep05(hm, n):
    H80 = hm.sum(0); se = np.sqrt(np.maximum(H80, 1))/(n*XB); seb = np.maximum(np.sqrt(hm.var(0)/n)/XB, 1/(n*XB)); return float(np.mean((seb/se)**2))
def lbins(li, ks, nl):
    lb = []
    for kk in (0, 1):
        m0 = ks == kk
        if not m0.any(): continue
        e = np.linspace(li[m0].min(), li[m0].max() + 1e-12, nl + 1)
        for b in range(nl):
            m_ = m0 & (li >= e[b]) & (li < e[b + 1])
            if m_.any(): lb.append([kk, round(float(li[m_].mean()), 8), int(m_.sum())])
    return lb
# ================================================================ stage 1: families
rng = np.random.default_rng(3463)
def statistics(hh, pr160, n):
    H = hh.sum(0).astype(float); o = {}
    for m, key in ((1, 'k025'), (2, 'k05'), (4, 'k1')):
        hk = hh.reshape(n, NF//m, m).sum(2).astype(np.float32); Hk = hk.sum(0, dtype=np.float64); sek = np.sqrt(np.maximum(Hk, 1))/(n*XF*m); sebk = np.maximum(np.sqrt(hk.var(0, dtype=np.float64)/n)/(XF*m), 1/(n*XF*m))
        o[key] = dict(independent=float(np.mean((sebk/sek)**2)), nbins=NF//m)
    hf = hh.astype(np.float32); cum_d = np.cumsum(H)/n; cum_p = np.cumsum(pr160)*XF; bs = {1: [], 2: [], 4: []}; sup = []
    for _ in range(BOOT):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(np.float32); hb = (w @ hf).astype(np.float64)
        for m in (1, 2, 4):
            Hm = agg(H, m); se_m = np.sqrt(np.maximum(Hm, 1))/(n*XF*m); bs[m].append(float(np.mean(((agg(hb, m) - Hm)/(n*XF*m)/se_m)**2)))
        sup.append(float(np.max(np.abs(np.cumsum(hb)/n - cum_d))))
    sup = np.array(sup); obs_sup = float(np.max(np.abs(cum_d - cum_p))); o['bootstrap'] = {}
    for m, key in ((1, 'k025'), (2, 'k05'), (4, 'k1')):
        Hm = agg(H, m); xb = XF*m; dens = Hm/(n*xb); se = np.sqrt(np.maximum(Hm, 1))/(n*xb); pm = aggm(pr160, m); ob = float(np.mean(((dens - pm)/se)**2)); b_ = np.array(bs[m])
        o[key]['chi2_ratios'] = ob; o['bootstrap'][key] = dict(mean=float(b_.mean()), q05=float(np.quantile(b_, 0.05)), q95=float(np.quantile(b_, 0.95)), p_below=float(np.mean(b_ <= ob)))
    o['bootstrap']['no_bins'] = dict(observed=obs_sup, mean=float(sup.mean()), q95=float(np.quantile(sup, 0.95)), p=float(np.mean(sup >= obs_sup)))
    return o
OUT1 = dict(study='III', xf=XF, nf=NF, xb=XB, nb=NB, bootstrap=BOOT, usp=B_usp.round(7).tolist(), sg=SG.tolist(), F_usp=F_usp.round(6).tolist(), m_usp=round(m_usp, 6), source=SRC, families={}, bands={})
HH_KEEP = {}
for f in ORDER:
    Ds, G = FAM[f]; R = {}; n_all = len(Ds)
    X = [theta(int(d), np.array(g))/math.pi for d, g in zip(Ds, G)]
    HH = np.stack([np.histogram(x[x < NF*XF], bins=np.arange(NF + 1)*XF)[0] for x in X]).astype(np.int16)
    L = np.log(np.abs(Ds).astype(float)/math.pi); S = (Ds < 0).astype(int)
    X1 = np.array([x[0] if len(x) and x[0] < 4 else np.nan for x in X])
    for sign, msk in (('all', np.ones(n_all, bool)), ('plus', S == 0), ('minus', S == 1)):
        idx = np.nonzero(msk)[0]; n = len(idx); hh = HH[idx]
        pr = mean_over('pr', L[idx], S[idx]); pr0 = mean_over('pr0', L[idx], S[idx]); wN = mean_over('w80', L[idx], S[idx])
        H = hh.sum(0).astype(float); H80 = agg(H, 2); pr80 = aggm(pr, 2); pr080 = aggm(pr0, 2)
        dens = H80/(n*XB); se = np.sqrt(np.maximum(H80, 1))/(n*XB)
        chi = lambda m_: float(np.nansum(((dens - m_)/se)**2)); d0 = lambda v: float(np.mean(v[:10]))
        li = L[idx]; lb = lbins(li, S[idx], NL)
        x1 = X1[idx]; x1 = x1[np.isfinite(x1)]; Fx1 = np.array([np.mean(x1 <= sg) for sg in SG]); N2 = 0.5*float(li.mean()); F2n, m2n = first_zero_model(N2)
        Ne = n_eff(d0(pr80)); Fef, mef = first_zero_model(Ne)
        Q = dict(n=n, mean_ln_D=float(np.mean(np.log(np.abs(Ds[idx]).astype(float)))), l_min=float(li.min()), l_max=float(li.max()), hist_0025=H.astype(int).tolist(),
                 prediction_0025=pr.round(7).tolist(), prediction_without_AD_0025=pr0.round(7).tolist(), usp2n_005=[None if not np.isfinite(v) else round(float(v), 6) for v in wN],
                 chi2_005_ratios=chi(pr80)/NB, chi2_005_without_AD=chi(pr080)/NB, chi2_005_usp=chi(B_usp)/NB, chi2_005_usp2n=chi(wN)/max(int(np.sum(np.isfinite(wN))), 1),
                 density_0_05=float(H[:20].sum()/(n*0.5)), density_0_05_se=float(math.sqrt(H[:20].sum())/(n*0.5)), density_0_05_ratios=d0(pr80), density_0_05_usp=d0(B_usp),
                 zeros_to_4=float(H.sum()/n), zeros_to_4_ratios=float(pr.sum()*XF),
                 x1_mean=float(np.mean(x1)), x1_se=float(np.std(x1)/math.sqrt(len(x1))), x1_n=int(len(x1)), x1_cdf=Fx1.round(6).tolist(), N_half_l=N2, x1_usp2n=m2n,
                 ks_usp=float(np.max(np.abs(Fx1 - F_usp))), ks_usp2n=float(np.max(np.abs(Fx1 - F2n))), ks_critical=float(1.36/math.sqrt(len(x1))),
                 N_eff=Ne, x1_usp2n_eff=mef, x1_cdf_usp2n_eff=Fef.round(6).tolist(), ks_usp2n_eff=float(np.max(np.abs(Fx1 - Fef))))
        Q.update(statistics(hh, pr, n)); R[sign] = Q
        print('%-5s %-5s n %6d chi2/bin %.3f (indep. %.3f, p %.3f) usp %.1f | [0,1/2] %.4f +- %.4f (%.4f) | x1 %.4f +- %.4f (eff %.4f, N_eff %.2f)'
              % (f, sign, n, Q['k05']['chi2_ratios'], Q['k05']['independent'], Q['bootstrap']['k05']['p_below'], Q['chi2_005_usp'], Q['density_0_05'], Q['density_0_05_se'], Q['density_0_05_ratios'], Q['x1_mean'], Q['x1_se'], mef, Ne), round(time.time() - t0), 's', flush=True)
    OUT1['families'][f] = R
    # bands of the size of family B in stage 1 (kept for the sequence of the random generator; stage 2 is the main test)
    if n_all >= 2*NPAS and f != 'C10':
        consecutive = f in ('B10', 'C100', 'D100', 'E10', 'E100'); nP = n_all//NPAS; order = np.argsort(np.abs(Ds), kind='stable'); hm_all = HH.reshape(n_all, NB, 2).sum(2); bands = []
        def rec(ii, typ):
            ii = np.sort(ii); m = len(ii); hm = hm_all[ii].astype(np.float32); H80 = hm.sum(0, dtype=np.float64); pr_ = mean_over('pr', L[ii], S[ii])
            h160 = HH[ii].astype(np.float32); H160 = h160.sum(0, dtype=np.float64)
            return dict(type=typ, n=m, D_abs=[int(np.abs(Ds[ii]).min()), int(np.abs(Ds[ii]).max())], chi2_005=chi05(H80, m, aggm(pr_, 2)), independent=indep05(hm, m))
        for j in range(nP): bands.append(rec(order[j*NPAS:(j + 1)*NPAS] if consecutive else rng.permutation(n_all)[:NPAS], 'consecutive' if consecutive else 'random'))
        for j in range(nP if f != 'B10' else 3*nP): bands.append(rec(rng.permutation(n_all)[:NPAS], 'random'))
        OUT1['bands'][f] = bands
    if f in ('B10', 'C100', 'D100'): HH_KEEP[f] = HH
    del X
OUT1['curve_one_D'] = [[round(L_, 3)] + [round(float(np.mean(predl(L_ - math.log(math.pi), k, xb=XB, nb=NB)[:10])), 5) for k in (0, 1)] for L_ in np.arange(6, 28.0001, 0.5)]   # density in [0, 1/2] for one D: ln|D|, D > 0, D < 0
json.dump(OUT1, open(os.path.join(OUT_DIR, 'reference_families%s.json' % SUFFIX), 'w')); print('stage 1 done', round(time.time() - t0), 's', flush=True)
if ONLY: sys.exit(0)
# ================================================================ stage 2: subsets (consecutive bands, random samples, samples stratified by D mod M)
NREP = {'B10': 81, 'C100': 300, 'D100': 30}; MODS = [8, 24, 120, 840]; rngm = np.random.default_rng(34630)
OUT2 = dict(study='III', subset_size=NPAS, moduli=MODS, families={})
for f in ('B10', 'C100', 'D100'):
    if f not in FAM: continue
    Ds, G = FAM[f]; n_all = len(Ds); HH = HH_KEEP[f]
    L = np.log(np.abs(Ds).astype(float)/math.pi); S = (Ds < 0).astype(int); hm_all = HH.reshape(n_all, NB, 2).sum(2)
    def rec(ii, typ):
        ii = np.sort(ii); m = len(ii); hm = hm_all[ii].astype(np.float32); H80 = hm.sum(0, dtype=np.float64); pr_ = mean_over('pr', L[ii], S[ii]); pm80 = aggm(pr_, 2)
        h40 = hm.reshape(m, NB//2, 2).sum(2); H40 = h40.sum(0, dtype=np.float64)
        dd = H40/(m*2*XB); e = np.sqrt(np.maximum(H40, 1))/(m*2*XB); c1 = float(np.mean(((dd - aggm(pr_, 4))/e)**2))
        r = (H80/(m*XB) - pm80)/(np.sqrt(np.maximum(H80, 1))/(m*XB)); c = FFT.dct(r, norm='ortho')**2
        return dict(type=typ, n=m, D_abs=[int(np.abs(Ds[ii]).min()), int(np.abs(Ds[ii]).max())], hist_005=H80.astype(int).tolist(), prediction_005=pm80.round(7).tolist(),
                    chi2_005=chi05(H80, m, pm80), chi2_01=c1, independent=indep05(hm, m), power=(c/NB).round(8).tolist(), low_k1_7=float(c[1:8].sum()/NB), high_k32_79=float(c[32:80].sum()/NB))
    order = np.argsort(np.abs(Ds), kind='stable'); nP = n_all//NPAS; nr = NREP[f]; sub = []
    for j in range(nP): sub.append(rec(order[j*NPAS:(j + 1)*NPAS], 'consecutive'))
    for j in range(nr): sub.append(rec(rngm.permutation(n_all)[:NPAS], 'random'))
    for M in MODS:
        tr = S*M + (Ds % M); uq, inv = np.unique(tr, return_inverse=True); Nc = np.bincount(inv); t = NPAS*Nc/n_all; fl = np.floor(t).astype(int)
        zb = NPAS - fl.sum(); fl[np.argsort(-(t - fl), kind='stable')[:zb]] += 1
        cl = [np.nonzero(inv == c)[0] for c in range(len(uq))]
        for j in range(nr):
            ii = np.concatenate([rngm.choice(cl[c], fl[c], replace=False) for c in range(len(uq)) if fl[c] > 0]); sub.append(rec(ii, 'mod%d' % M))
    OUT2['families'][f] = dict(n=n_all, subsets=sub)
    for typ in ['consecutive', 'random'] + ['mod%d' % M for M in MODS]:
        v = np.array([q['chi2_005'] for q in sub if q['type'] == typ]); lo = np.array([q['low_k1_7'] for q in sub if q['type'] == typ])
        print('%-5s %-11s %3d chi2/bin %.3f +- %.3f | low k 1-7 %.4f +- %.4f' % (f, typ, len(v), v.mean(), v.std(ddof=1)/math.sqrt(len(v)), lo.mean(), lo.std(ddof=1)/math.sqrt(len(lo))), flush=True)
json.dump(OUT2, open(os.path.join(OUT_DIR, 'reference_subsets.json'), 'w')); print('stage 2 done', round(time.time() - t0), 's', flush=True)
# ================================================================ stage 3: pieces
MS = [1, 3, 10, 30, 100, 300, 1000, NPAS]; NREPK = {'B10': 60, 'C100': 150}; rngk = np.random.default_rng(346301); OUT3 = dict(study='III', subset_size=NPAS, m=MS, families={})
for f in ('B10', 'C100'):
    if f not in FAM: continue
    Ds, G = FAM[f]; n_all = len(Ds); HH = HH_KEEP[f]
    L = np.log(np.abs(Ds).astype(float)/math.pi); S = (Ds < 0).astype(int); hm_all = HH.reshape(n_all, NB, 2).sum(2); order = np.argsort(np.abs(Ds), kind='stable')
    dD = float(np.abs(Ds).max() - np.abs(Ds).min())/n_all*NPAS; lm = float(L.mean()); R = dict(n=n_all, l=lm, stretch_D=dD, m={})
    for m in MS:
        ln_ = NPAS//m; rem = NPAS - ln_*m; res = []
        for _ in range(NREPK[f]):
            if m == 1: st = [int(rngk.integers(0, n_all - NPAS))]; ii = order[st[0]:st[0] + NPAS]
            elif m == NPAS: ii = rngk.permutation(n_all)[:NPAS]
            else:
                dl = [ln_ + (1 if j < rem else 0) for j in range(m)]; st = np.sort(rngk.choice(n_all//max(dl), m, replace=False))*max(dl); ii = np.concatenate([order[s_:s_ + d_] for s_, d_ in zip(st, dl)])
            ii = np.sort(ii); mm = len(ii); hm = hm_all[ii].astype(np.float64); H80 = hm.sum(0); pm80 = aggm(mean_over('pr', L[ii], S[ii]), 2)
            dd = H80/(mm*XB); se = np.sqrt(np.maximum(H80, 1))/(mm*XB); c = FFT.dct((dd - pm80)/se, norm='ortho')**2
            res.append(dict(chi=float(c.sum()/80), bands=[float(c[a:b].sum()/80) for a, b in ((0, 1), (1, 4), (4, 8), (8, 16), (16, 32), (32, 80))], spec=c))
        sp_ = np.array([v['spec'] for v in res]); ch = np.array([v['chi'] for v in res])
        R['m'][str(m)] = dict(stretch_D=dD/m, u_c=math.log(dD/m)/lm if dD/m > 1 else 0.0, chi2_005=float(ch.mean()), chi2_005_se=float(ch.std(ddof=1)/math.sqrt(len(ch))), power=(sp_.mean(0)/80).round(8).tolist(),
                              bands=np.array([v['bands'] for v in res]).mean(0).round(6).tolist())
        print('%-5s m %5d stretch %9.1f u_c %.2f chi2 %.3f low k 1-7 %.4f' % (f, m, dD/m, R['m'][str(m)]['u_c'], ch.mean(), sum(R['m'][str(m)]['bands'][1:3])), flush=True)
    OUT3['families'][f] = R
json.dump(OUT3, open(os.path.join(OUT_DIR, 'reference_pieces.json'), 'w')); print('done', round(time.time() - t0), 's', flush=True)
