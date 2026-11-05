#!/usr/bin/env python3
"""
Low-lying zeros of quadratic Dirichlet L-functions, study III: second-order coupling of the modes (Section 4.6).

usage:  python3 -I intermodulation.py ZEROS_ROOT OUT_DIR          (about 2 minutes, 4 GB RAM; writes reference_intermodulation.json)

y_ik is the contribution of L-function i to the cosine coefficient c_k (k = 1 ... 7) with the Pearson error, exactly as in exact_layers.py.
The modes k and 8 - k of one L-function are correlated (exact_layers.py, coupling of modes).  Here: is this coupling of second order in the
characters?  The regression of y_ik on 1, chi_D(p), chi_D(p)^2 for all primes p <= 300 (base) is extended by chi_D(n) of squarefree composite
n <= 1000 (n with at least two prime factors, chi_D(n) = product of chi_D(p) over p | n), grouped by ln n (windows 1.8-4, 4-5, 5-6, 6-6.9).
Gain = increase of the explained share of the variance of y_ik for every mode k.  Null model: the same composite features permuted over the
L-functions (same distribution, no relation to the zeros), 4 permutations.  If the modes 8 - k are second order in the characters that carry
the modes k (ln p ~ k l/8), the products chi_D(p)chi_D(q) of the small primes raise the explained share of the modes 6-7 and hardly of 1-4.
Also: layer 0 of a sample balanced on chi_D(n) of all n <= 1000 (primes and composites; residual variance as in exact_layers.py).
Normal equations are accumulated in blocks of rows; the permuted features need only new cross blocks.
Filters in the frequency of the primes (Sections 4.3-4.4): the prime p acts at u_p = ln p / l, i.e. at the cosine mode k_p = 8 ln p / l.  A stretch of width Delta
in |D| is a low-pass filter up to u_c = ln Delta / l (k_c = 8 u_c); written here are Delta, u_c, k_c.  The stratification by (sign, D mod 840) is a comb filter:
its layer 0 is that of the regression on the indicators of all classes (within-class variance), decomposed into the prime lines p = 2, 3, 5, 7 (chi_D(p) and
chi_D(p)^2), the sign, and the products chi_D(n) of these primes (n = 6, 10, ..., 210) - the intermodulation lines - also multiplied by the sign.
"""
import sys, os, json, math, time, numpy as np, mpmath as mp, scipy.special as sp, scipy.fft as FFT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
t0 = time.time(); ROOT, OUT_DIR = sys.argv[1], sys.argv[2]; os.makedirs(OUT_DIR, exist_ok=True)
DT = 0.002; TG = np.round(np.arange(0, 8.0 + 1e-9, DT), 4); NT = len(TG); XF = 0.025; NF = 160; XB = 0.05; NB = 80; LB = 0.002; N = 6074
FAMS = ('B10', 'C100', 'D100'); PBASE = 300; NMAX = 1000; WIN = [math.log(6), 4.0, 5.0, 6.0, math.log(NMAX) + 1e-9]; NPERM = 4; SEED = 346810; BLK = 20000
RF = RZ.families(ROOT)
# ---------------------------------------------------------------- the ratios conjecture on a grid in t (as compute_families.py)
Mp = 200000; s = np.ones(Mp + 1, bool); s[:2] = False
for p in range(2, int(Mp**0.5) + 1):
    if s[p]: s[p*p::p] = False
PRM = np.nonzero(s)[0].astype(float); lp = np.log(PRM)
A1 = np.empty(NT, complex); A2 = np.empty(NT, complex)
for i, tt in enumerate(TG):
    z = np.exp(2j*tt*lp); A1[i] = np.exp(np.sum(np.log(1 - (z - 1)/(PRM*PRM - 1)))); A2[i] = np.sum(lp/((PRM + 1)*(PRM*z - 1)))
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
CACHE = {}
def node(k, j):
    if (k, j) not in CACHE: CACHE[(k, j)] = predl(j*LB, k).reshape(NB, 2).mean(1)
    return CACHE[(k, j)]
A = FFT.dct(np.eye(NB), norm='ortho', axis=0)[1:8]                  # rows k = 1 ... 7 of the orthonormal cosine transform
TABS = {}
def kron(D, p):
    if p == 2: m = np.mod(D, 8); return np.where(D % 2 == 0, 0, np.where((m == 1) | (m == 7), 1, -1)).astype(np.float64)
    if p not in TABS: TABS[p] = np.array([0] + [1 if pow(r, (p - 1)//2, p) == 1 else -1 for r in range(1, p)], dtype=np.float64)
    return TABS[p][np.mod(D, p)]
def primes(P): return [q for q in range(2, P + 1) if all(q % r for r in range(2, int(math.isqrt(q)) + 1))]
OUT = dict(study='III', subset_size=N, base_primes_up_to=PBASE, composites_up_to=NMAX, windows_ln_n=WIN, permutations=NPERM, seed=SEED, families={})
for f in FAMS:
    Ds, G = [], []
    for D, g in RZ.zeros(f, ROOT): Ds.append(D); G.append(g)
    Ds = np.array(Ds, dtype=np.int64); n = len(Ds); X = [theta(int(d), np.array(g))/math.pi for d, g in zip(Ds, G)]
    hm = np.stack([np.histogram(x[x < NB*XB], bins=np.arange(NB + 1)*XB)[0] for x in X]).astype(np.float64); del X, G
    L = np.log(np.abs(Ds).astype(float)/math.pi); S = (Ds < 0).astype(int); order = np.argsort(np.abs(Ds), kind='stable')
    P = np.empty((n, NB))                                          # prediction of the ratios conjecture for every D (nodes 0.002 in l, linear)
    for k in (0, 1):
        idx = np.nonzero(S == k)[0]; j = np.floor(L[idx]/LB).astype(np.int64); u = L[idx]/LB - j
        for jj in np.unique(j):
            q = idx[j == jj]; uu = u[j == jj][:, None]; P[q] = (1 - uu)*node(k, int(jj))[None, :] + uu*node(k, int(jj) + 1)[None, :]
    Dv = hm/XB - P                                                 # deviation of the density of one L-function from its prediction
    seP = np.sqrt(N*XB*P.mean(0))/(N*XB)                           # Pearson: error of a subset of N from the expected count of the family prediction
    Y = (Dv/seP[None, :]) @ A.T                                    # y_ik: contribution of L-function i to the coefficient c_k (layer 0 is linear in it)
    ybar = Y.mean(0); S2 = Y.var(0, ddof=1); del hm, P, Dv
    PR = primes(NMAX); pos = {p: i for i, p in enumerate(PR)}; npb = sum(1 for p in PR if p <= PBASE)
    comp = []
    for m in range(6, NMAX + 1):
        fs, x = [], m
        for p in PR:
            if p*p > x: break
            if x % p == 0:
                x //= p; fs.append(p)
                if x % p == 0: fs = None; break
        if fs is None: continue
        if x > 1: fs.append(x)
        if len(fs) >= 2: comp.append((m, fs))
    lnc = np.log(np.array([m for m, _ in comp], float)); nc = len(comp)
    CH = np.stack([kron(Ds, p) for p in PR], 1).astype(np.int8)                     # chi_D(p), p <= 1000
    CF = np.stack([np.prod(CH[:, [pos[p] for p in fs]].astype(np.int16), 1) for _, fs in comp], 1).astype(np.int8)   # chi_D(n), composite n <= 1000
    def feat(a0, a1, rows=None):
        ch = CH[a0:a1].astype(np.float64); cf = (CF[a0:a1] if rows is None else CF[rows[a0:a1]]).astype(np.float64)
        return np.hstack([np.ones((a1 - a0, 1)), ch[:, :npb], ch[:, :npb]**2]), np.hstack([ch[:, npb:], ch[:, npb:]**2]), cf
    q0, q1 = 1 + 2*npb, 2*(len(PR) - npb)
    G = np.zeros((q0 + q1 + nc, q0 + q1 + nc)); bY = np.zeros((q0 + q1 + nc, 7))
    for a0 in range(0, n, BLK):
        F = np.hstack(feat(a0, min(n, a0 + BLK))); G += F.T @ F; bY += F.T @ Y[a0:a0 + BLK]
    YY = (Y*Y).sum(0)
    def r2(cols, Gm=None, bm=None):
        Gm = G if Gm is None else Gm; bm = bY if bm is None else bm; c = np.array(cols); g = Gm[np.ix_(c, c)]; bb = bm[c]
        coef = np.linalg.lstsq(g, bb, rcond=None)[0]; s2r = np.maximum(YY - (coef*bb).sum(0), 0)/(n - 1)
        return 1 - s2r/S2, float((ybar**2 + s2r/N*(1 - N/n)).sum()/80)
    I0 = list(range(q0)); I1 = list(range(q0, q0 + q1)); IC = [q0 + q1 + j for j in range(nc)]
    win = [[IC[j] for j in range(nc) if WIN[w] <= lnc[j] < WIN[w + 1]] for w in range(len(WIN) - 1)]
    b0, e0 = r2(I0); R = dict(n=n, l=float(L.mean()), mode_scale_ln_p=(np.arange(1, 8)*float(L.mean())/8).tolist(), composites=nc, base_explained=b0.tolist(), base_layer0=e0, windows=[])
    rng = np.random.default_rng([SEED, FAMS.index(f)])
    for w, cols in enumerate(win):
        bw, ew = r2(I0 + cols); nul = []
        for r in range(NPERM):
            perm = rng.permutation(n); Gp = G.copy(); bp = bY.copy(); c = np.array(cols); jj = [x - q0 - q1 for x in cols]
            X0 = np.zeros((q0, len(cols))); XY = np.zeros((len(cols), 7))
            for a0 in range(0, n, BLK):
                a1 = min(n, a0 + BLK); f0 = feat(a0, a1)[0]; cp = CF[perm[a0:a1]][:, jj].astype(np.float64); X0 += f0.T @ cp; XY += cp.T @ Y[a0:a1]
            Gp[np.ix_(I0, c)] = X0; Gp[np.ix_(c, I0)] = X0.T; bp[c] = XY        # composite block itself is unchanged by the permutation
            nul.append(r2(I0 + cols, Gp, bp)[0] - b0)
        nul = np.array(nul); gain = bw - b0
        R['windows'].append(dict(ln_n=[WIN[w], WIN[w + 1]], features=len(cols), gain=gain.tolist(), null_mean=nul.mean(0).tolist(), null_sd=nul.std(0, ddof=1).tolist(), layer0=ew))
        print('%-5s composites ln n %.1f-%.1f (%3d): gain %s  null %s' % (f, WIN[w], WIN[w + 1], len(cols), np.round(gain, 4).tolist(), np.round(nul.mean(0), 4).tolist()), round(time.time() - t0), 's', flush=True)
    ba, ea = r2(I0 + IC); bp1, ep1 = r2(I0 + I1); bpa, epa = r2(I0 + I1 + IC)
    R.update(all_composites=dict(explained=ba.tolist(), layer0=ea), primes_1000=dict(explained=bp1.tolist(), layer0=ep1), primes_and_composites_1000=dict(explained=bpa.tolist(), layer0=epa),
             random_layer0=float((ybar**2 + S2/N*(1 - N/n)).sum()/80))
    lo = R['windows'][0]; g = np.array(lo['gain']); R['ratio_modes_67_to_14'] = float(g[5:7].mean()/g[0:4].mean())
    # ---- filters: stretch (low-pass up to u_c) and D mod 840 (comb at the primes 2, 3, 5, 7 and their products)
    Dl = float(np.abs(Ds).max() - np.abs(Ds).min())/n*N; lm = float(L.mean()); R['stretch_width'] = Dl; R['u_c'] = math.log(Dl)/lm; R['k_c'] = 8*math.log(Dl)/lm
    cls = (Ds < 0).astype(np.int64)*840 + np.mod(Ds, 840); uq, ic = np.unique(cls, return_inverse=True); cnt = np.bincount(ic).astype(float); sums = np.zeros((len(uq), 7)); np.add.at(sums, ic, Y)
    within = np.maximum(YY - (sums**2/cnt[:, None]).sum(0), 0)/(n - 1); e840 = float((ybar**2 + within/N*(1 - N/n)).sum()/80)
    P7 = (2, 3, 5, 7); c7 = np.stack([kron(Ds, p) for p in P7], 1); sg = (Ds < 0).astype(np.float64)[:, None]; PRD = (6, 10, 14, 15, 21, 35, 30, 42, 70, 105, 210)
    cpr = np.stack([np.prod([kron(Ds, p) for p in P7 if m % p == 0], 0) for m in PRD], 1); one = np.ones((n, 1))
    def e0reg(F_):
        G_ = F_.T @ F_; b_ = F_.T @ Y; cf = np.linalg.lstsq(G_, b_, rcond=None)[0]; rs = np.maximum(YY - (cf*b_).sum(0), 0)/(n - 1); return float((ybar**2 + rs/N*(1 - N/n)).sum()/80), (1 - rs/S2).tolist()
    steps = []
    for nm_, F_ in (('primes 2, 3, 5, 7', np.hstack([one, c7, c7**2])), ('+ sign', np.hstack([one, c7, c7**2, sg])), ('+ products of 2, 3, 5, 7', np.hstack([one, c7, c7**2, sg, cpr])),
                    ('+ products and primes times sign', np.hstack([one, c7, c7**2, sg, cpr, cpr*sg, c7*sg, c7**2*sg]))):
        e_, x_ = e0reg(F_); steps.append(dict(features=nm_, layer0=e_, explained=x_))
    R['filter840'] = dict(classes=int(len(uq)), stratification_layer0=e840, stratification_explained=(1 - within/S2).tolist(), steps=steps, mode_of_prime={str(p): 8*math.log(p)/lm for p in P7})
    print('%-5s filters: stretch %.0f, u_c %.3f, k_c %.2f; D mod 840 (%d classes) layer 0 %.5f; steps %s' % (f, Dl, R['u_c'], R['k_c'], len(uq), e840, ' / '.join('%.5f' % q['layer0'] for q in steps)), round(time.time() - t0), 's', flush=True)
    del c7, cpr, sg, one
    OUT['families'][f] = R
    print('%-5s base p <= %d: explained %s layer 0 %.5f; all composites %.5f; primes <= %d %.5f; primes and composites %.5f; ratio 6-7 / 1-4 (ln n below 4) %.1f'
          % (f, PBASE, np.round(b0, 3).tolist(), e0, ea, NMAX, ep1, epa, R['ratio_modes_67_to_14']), round(time.time() - t0), 's', flush=True)
    del CH, CF, G
json.dump(OUT, open(os.path.join(OUT_DIR, 'reference_intermodulation.json'), 'w')); print('done', round(time.time() - t0), 's', flush=True)
