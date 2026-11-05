#!/usr/bin/env python3
"""
Low-lying zeros of quadratic Dirichlet L-functions, study III: the test with pieces (Section 4.3, Figure 5 right) with the error of every point.

usage:  python3 -I pieces_repeat.py ZEROS_ROOT OUT_DIR          (about 4 minutes, 3 GB RAM; writes reference_pieces_repeat.json)

A subset of 6 074 L is assembled from m consecutive pieces (sorted by |D|) at random positions; m = 1 is a consecutive band, m = 6 074 a random sample.
For every subset the 80 normalised deviations (data - prediction of the ratios conjecture)/error in bins of 0.05 are decomposed by the orthonormal cosine
transform and the power at k = 1 ... 7 is summed (as in compute_families.py, stage 3).
  (1) first run: the same subsets as stage 3 of compute_families.py (one random generator, seed 346301, first B10 with 60, then C100 with 150 subsets per m);
      the mean power is that of reference_pieces.json, and the error of the mean is added (standard deviation of the subsets / sqrt(number)).
  (2) repetition: a new random generator for each family (seed 346802), 400 subsets per m, m = 1 ... 6074 including m = 2 000.
  (3) B10, three further seeds (346803 ... 346805) with 400 subsets for m = 1 000 and m = 6 074: the scatter between independent runs.
With 60 subsets per point the error of the mean power in B10 is 0.003-0.005, about a tenth of the value; neighbouring points can then differ by more than the
true curve does. The repetition shows the curve without such steps.
"""
import sys, os, json, math, time, numpy as np, mpmath as mp, scipy.special as sp, scipy.fft as FFT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
t0 = time.time(); ROOT, OUT_DIR = sys.argv[1], sys.argv[2]; os.makedirs(OUT_DIR, exist_ok=True)
DT = 0.002; TG = np.round(np.arange(0, 8.0 + 1e-9, DT), 4); NT = len(TG); XF = 0.025; NF = 160; XB = 0.05; NB = 80; LB = 0.002; NPAS = 6074
FAM = {}; RF = RZ.families(ROOT)
for f in ('B10', 'C100'):
    if f not in RF: raise SystemExit('family %s not found under %s' % (f, ROOT))
    Ds, G = [], []
    for D, g in RZ.zeros(f, ROOT): Ds.append(D); G.append(g)
    FAM[f] = (np.array(Ds, dtype=np.int64), G)
print('families', {f: len(FAM[f][0]) for f in FAM}, round(time.time() - t0), 's', flush=True)
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
Z1 = ZL + A2; Z2 = {k: GR[k]*ZR*A1 for k in (0, 1)}
def theta(d, t): k = 0 if d > 0 else 1; return 0.5*np.asarray(t)*math.log(abs(d)/math.pi) + np.imag(sp.loggamma(0.25 + 0.5*k + 0.5j*np.asarray(t)))
def predl(l, k):
    c = np.real(Z1 - np.exp(-1j*TG*l)*Z2[k])/math.pi; c[0] = 3*c[1] - 3*c[2] + c[3]
    C = np.concatenate([[0], np.cumsum(0.5*(c[1:] + c[:-1])*DT)])
    th = (0.5*TG*l + IMLG[k])/math.pi; te = np.interp(np.arange(NF + 1)*XF, th, TG)
    return 1 + np.diff(np.interp(te, TG, C))/XF
aggm = lambda a, m: a.reshape(-1, m).mean(1)
CACHE = {}
def node(k, j):
    if (k, j) not in CACHE: CACHE[(k, j)] = predl(j*LB, k)
    return CACHE[(k, j)]
def mean_pred(L, S):
    tot = None
    for k in (0, 1):
        m = S == k
        if not m.any(): continue
        li = L[m]; j = np.floor(li/LB).astype(np.int64); u = li/LB - j
        uj, inv = np.unique(j, return_inverse=True); cnt = np.bincount(inv).astype(float); su = np.bincount(inv, weights=u)
        for q, jj in enumerate(uj):
            v = (cnt[q] - su[q])*node(k, int(jj)) + su[q]*node(k, int(jj) + 1); tot = v if tot is None else tot + v
    return tot/len(L)
PREP = {}
for f, (Ds, G) in FAM.items():
    n_all = len(Ds); X = [theta(int(d), np.array(g))/math.pi for d, g in zip(Ds, G)]
    HH = np.stack([np.histogram(x[x < NF*XF], bins=np.arange(NF + 1)*XF)[0] for x in X]).astype(np.int16)
    L = np.log(np.abs(Ds).astype(float)/math.pi)
    PREP[f] = dict(n=n_all, hm=HH.reshape(n_all, NB, 2).sum(2), L=L, S=(Ds < 0).astype(int), order=np.argsort(np.abs(Ds), kind='stable'),
                   dD=float(np.abs(Ds).max() - np.abs(Ds).min())/n_all*NPAS, l=float(L.mean()))
    del HH, X
def subset(f, m, rng):
    Q = PREP[f]; n_all = Q['n']; ln_ = NPAS//m; rem = NPAS - ln_*m
    if m == 1: st = int(rng.integers(0, n_all - NPAS)); ii = Q['order'][st:st + NPAS]
    elif m == NPAS: ii = rng.permutation(n_all)[:NPAS]
    else:
        dl = [ln_ + (1 if j < rem else 0) for j in range(m)]; st = np.sort(rng.choice(n_all//max(dl), m, replace=False))*max(dl); ii = np.concatenate([Q['order'][a:a + d] for a, d in zip(st, dl)])
    ii = np.sort(ii); mm = len(ii); hm = Q['hm'][ii].astype(np.float64); H80 = hm.sum(0); pm80 = aggm(mean_pred(Q['L'][ii], Q['S'][ii]), 2)
    dd = H80/(mm*XB); se = np.sqrt(np.maximum(H80, 1))/(mm*XB); c = FFT.dct((dd - pm80)/se, norm='ortho')**2
    return [float(c[a:b].sum()/80) for a, b in ((0, 1), (1, 4), (4, 8), (8, 16), (16, 32), (32, 80))]
def summary(f, m, V):
    V = np.array(V); low = V[:, 1] + V[:, 2]; dD = PREP[f]['dD']/m
    return dict(stretch_D=dD, u_c=math.log(dD)/PREP[f]['l'] if dD > 1 else 0.0, n_subsets=len(low), power_k1_7=float(low.mean()), power_k1_7_se=float(low.std(ddof=1)/math.sqrt(len(low))),
                power_k1_7_sd=float(low.std(ddof=1)), bands=V.mean(0).round(6).tolist())
MS1 = [1, 3, 10, 30, 100, 300, 1000, NPAS]; MS2 = [1, 3, 10, 30, 100, 300, 1000, 2000, NPAS]
OUT = dict(study='III', subset_size=NPAS, first_run=dict(seed=346301, subsets={'B10': 60, 'C100': 150}, families={}), repetition=dict(seed=346802, subsets=400, families={}), seeds_B10={})
rng1 = np.random.default_rng(346301)
for f, nr in (('B10', 60), ('C100', 150)):
    R = {}
    for m in MS1:
        R[str(m)] = summary(f, m, [subset(f, m, rng1) for _ in range(nr)])
        print('first run   %-5s m %5d  power k 1-7 %.4f +- %.4f (%d subsets)' % (f, m, R[str(m)]['power_k1_7'], R[str(m)]['power_k1_7_se'], nr), flush=True)
    OUT['first_run']['families'][f] = dict(l=PREP[f]['l'], stretch_D=PREP[f]['dD'], m=R)
for f in ('B10', 'C100'):
    rng2 = np.random.default_rng(346802); R = {}
    for m in MS2:
        R[str(m)] = summary(f, m, [subset(f, m, rng2) for _ in range(400)])
        print('repetition  %-5s m %5d  power k 1-7 %.4f +- %.4f (400 subsets)' % (f, m, R[str(m)]['power_k1_7'], R[str(m)]['power_k1_7_se']), flush=True)
    OUT['repetition']['families'][f] = dict(l=PREP[f]['l'], stretch_D=PREP[f]['dD'], m=R)
for sd_ in (346803, 346804, 346805):
    rng3 = np.random.default_rng(sd_); OUT['seeds_B10'][str(sd_)] = {}
    for m in (1000, NPAS):
        r_ = summary('B10', m, [subset('B10', m, rng3) for _ in range(400)]); OUT['seeds_B10'][str(sd_)][str(m)] = dict(power_k1_7=r_['power_k1_7'], power_k1_7_se=r_['power_k1_7_se'])
        print('seed %d B10 m %5d  power k 1-7 %.4f +- %.4f' % (sd_, m, r_['power_k1_7'], r_['power_k1_7_se']), flush=True)
json.dump(OUT, open(os.path.join(OUT_DIR, 'reference_pieces_repeat.json'), 'w')); print('done', round(time.time() - t0), 's', flush=True)
