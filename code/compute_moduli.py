#!/usr/bin/env python3
"""
Study III, section 4.4: residue classes D mod M for 19 moduli and the sign, with an independent repetition (reference_moduli.json).
Subsets of 6 074 L-functions (the size of family B) in B10, C100 and D100: consecutive bands (sorted by |D|), random samples, and random samples stratified to the
composition of the whole family in the classes (sign, D mod M); for each subset the power of the cosine coefficients k = 1 ... 7 of the 80 normalised deviations of the
histogram of zeros (bins of 0.05) from the ratios conjecture (as compute_families.py, stage 2). The common part (unfolding, ratios conjecture, averaging over D) is that of
compute_families.py; the low-mode power of random samples and of consecutive bands of pass 1 reproduces study III v.34.63.

usage:  python3 -I compute_moduli.py ZEROS_ROOT reference_moduli.json        (about 8 minutes)
"""
import sys, os, json, math, time, numpy as np, mpmath as mp, scipy.special as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
t0 = time.time()
ROOT = sys.argv[1]; OUTP = sys.argv[2]
DT = 0.002; TG = np.round(np.arange(0, 8.0 + 1e-9, DT), 4); NT = len(TG); XF = 0.025; NF = 160; XB = 0.05; NB = 80; NL = 24; BOOT = 200; LB = 0.002
NPAS = 6074                                    # size of a subset = family B (|D| = 10^4 ... 2·10^4)
ORDER_ALL = ['B10', 'C100', 'D100']
FAM = {}; SRC = {}; RS = RZ.families(ROOT)
for jm in ORDER_ALL:
    if jm not in RS: continue
    Ds, G = [], []
    for D, g in RZ.zeros(jm, ROOT): Ds.append(D); G.append(g)
    FAM[jm] = (np.array(Ds, dtype=np.int64), G); SRC[jm] = RS[jm][0]
ORDER = [j for j in ORDER_ALL if j in FAM]
print('families', {j: len(FAM[j][0]) for j in ORDER}, round(time.time() - t0), 's', flush=True)
# ---------------------------------------------------------------- the ratios conjecture on a grid in t (as compute_families.py)
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
print('grid', NT, round(time.time() - t0), 's', flush=True)
def theta(d, t): k = 0 if d > 0 else 1; return 0.5*np.asarray(t)*math.log(abs(d)/math.pi) + np.imag(sp.loggamma(0.25 + 0.5*k + 0.5j*np.asarray(t)))
def predl(l, k, ar=True, xb=XF, nb=NF):
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
# ---------------------------------------------------------------- mean of the prediction over a set of D: nodes in l every LB, linear interpolation (error ~ f''·LB²/8)
CACHE = {}
FUN = {'pr': lambda l, k: predl(l, k), 'pr0': lambda l, k: predl(l, k, False), 'w80': lambda l, k: binavg(lambda xx: W_N(xx, 0.5*l))}
def node(fn, k, j):
    key = (fn, k, j)
    if key not in CACHE: CACHE[key] = FUN[fn](j*LB, k)
    return CACHE[key]
def mean_over(fn, L, S):
    """mean of FUN[fn](l_D, k_D) over all D (arrays L, S)"""
    tot = None
    for k in (0, 1):
        m = S == k
        if not m.any(): continue
        li = L[m]; j = np.floor(li/LB).astype(np.int64); u = li/LB - j
        uj, inv = np.unique(j, return_inverse=True); cnt = np.bincount(inv).astype(float); su = np.bincount(inv, weights=u)
        for q, jj in enumerate(uj):
            f0 = node(fn, k, int(jj)); f1 = node(fn, k, int(jj) + 1); v = (cnt[q] - su[q])*f0 + su[q]*f1
            tot = v if tot is None else tot + v
    return tot/len(L)
def chi05(H80, n, pm80):
    dens = H80/(n*XB); se = np.sqrt(np.maximum(H80, 1))/(n*XB); return float(np.mean(((dens - pm80)/se)**2))
# ---------------------------------------------------------------- subsets of 6 074 L: consecutive bands, random samples and samples stratified by (sign, D mod M)
# Three passes with their own random generators (as three separate runs): pass 1 the moduli of study III v.34.63 (seed 34639) and new moduli (seed 34680, 81 / 100 / 30 samples),
# pass 2 an independent repetition with more samples and the sign-only check M = 1 (seed 34681, 300 / 300 / 100), pass 3 the modulus 131 (seed 34683, 300 / 300 / 100).
PASSES = [dict(name='pass1', nrep={'B10': 81, 'C100': 100, 'D100': 30}, groups=[([130, 260, 360, 369, 520, 800, 840], 34639), ([8, 16, 3, 9, 5, 7, 13, 41, 24, 40, 120], 34680)], random_seed_group=0, bands=True),
          dict(name='pass2', nrep={'B10': 300, 'C100': 300, 'D100': 100}, groups=[([1, 8, 16, 3, 7, 41, 360, 520, 840], 34681)], random_seed_group=0, bands=False),
          dict(name='pass3', nrep={'B10': 300, 'C100': 300, 'D100': 100}, groups=[([131], 34683)], random_seed_group=0, bands=False)]
RNG = {(p['name'], g[1]): np.random.default_rng(g[1]) for p in PASSES for g in p['groups']}
import scipy.fft as FFT
OUT = dict(study='III', version='34.68', subset_size=NPAS, passes=[dict(name=p['name'], nrep=p['nrep'], groups=[[g[0], g[1]] for g in p['groups']]) for p in PASSES], families={})
for jm in ORDER:
    Ds, G = FAM[jm]; n_all = len(Ds)
    X = [theta(int(d), np.array(g))/math.pi for d, g in zip(Ds, G)]
    HH = np.stack([np.histogram(x[x < NF*XF], bins=np.arange(NF + 1)*XF)[0] for x in X]).astype(np.int16)
    L = np.log(np.abs(Ds).astype(float)/math.pi); S = (Ds < 0).astype(int); hm_all = HH.reshape(n_all, NB, 2).sum(2)
    def record(ii):
        ii = np.sort(ii); m = len(ii); hm = hm_all[ii].astype(np.float32); H80 = hm.sum(0, dtype=np.float64); pr_ = mean_over('pr', L[ii], S[ii]); pm80 = aggm(pr_, 2)
        r = (H80/(m*XB) - pm80)/(np.sqrt(np.maximum(H80, 1))/(m*XB)); low = float((FFT.dct(r, norm='ortho')**2)[1:8].sum()/80)
        return dict(chi=chi05(H80, m, pm80), low=low)
    order = np.argsort(np.abs(Ds), kind='stable'); nP = n_all//NPAS; F = dict(n=n_all, types={})
    for p in PASSES:
        nr = p['nrep'][jm]; rg0 = RNG[(p['name'], p['groups'][p['random_seed_group']][1])]
        if p['bands']: F['types'].setdefault('consecutive', {})[p['name']] = [record(order[j*NPAS:(j + 1)*NPAS]) for j in range(nP)]
        F['types'].setdefault('random', {})[p['name']] = [record(rg0.permutation(n_all)[:NPAS]) for j in range(nr)]
        for mods, seed in p['groups']:
            rg = RNG[(p['name'], seed)]
            for M in mods:
                tr = S*M + (Ds % M); uq, inv = np.unique(tr, return_inverse=True); Nc = np.bincount(inv); t = NPAS*Nc/n_all; fl = np.floor(t).astype(int)
                zb = NPAS - fl.sum(); fl[np.argsort(-(t - fl), kind='stable')[:zb]] += 1; cl = [np.nonzero(inv == c)[0] for c in range(len(uq))]
                F['types'].setdefault('mod%d' % M, {})[p['name']] = [record(np.concatenate([rg.choice(cl[c], fl[c], replace=False) for c in range(len(uq)) if fl[c] > 0])) for j in range(nr)]
        print(jm, p['name'], round(time.time() - t0), 's', flush=True)
    for typ, v in F['types'].items():
        lo = np.array([q['low'] for ps_ in v.values() for q in ps_]); F['types'][typ] = dict(passes={k: [round(q['low'], 8) for q in ps_] for k, ps_ in v.items()}, chi={k: [round(q['chi'], 6) for q in ps_] for k, ps_ in v.items()},
                                                                                          n=int(len(lo)), low_mean=float(lo.mean()), low_se=float(lo.std(ddof=1)/math.sqrt(len(lo))) if len(lo) > 1 else None)
        print('%-5s %-12s n %4d  low k 1-7 %.4f +- %.4f' % (jm, typ, len(lo), lo.mean(), F['types'][typ]['low_se'] or 0), flush=True)
    OUT['families'][jm] = F; del HH, X
json.dump(OUT, open(OUTP, 'w')); print('done', round(time.time() - t0), 's', flush=True)
