#!/usr/bin/env python3
"""
Low-lying zeros of quadratic Dirichlet L-functions, study III, Section 4.7: every number of Sections 4.3 and 4.4 that was obtained from random
subsets, recomputed by layers (the principle of Section 4.6) and set beside its original value.

usage:  python3 -I layered_recomputation.py ZEROS_ROOT DATA_DIR OUT_DIR      (about 25 minutes on two cores, 4 GB RAM; writes reference_recomputation.json)
        DATA_DIR holds the original results reference_subsets.json, reference_moduli.json, reference_pieces_repeat.json and reference_characters.json
        python3 -I layered_recomputation.py ZEROS_ROOT DATA_DIR OUT_DIR --families E100      (Section 4.8, about 15 minutes; writes
        reference_recomputation_E100.json: the same designs for a family without original values; the remaining shares W100 of the stratifications
        are computed here from the exact stratified variance of the means of chi_D(p), p <= 100, and the line uses them)

The statistics are those of Sections 4.3-4.4 for a subset of N = 6 074 L-functions: chi^2 per bin (bins 0.05 and 0.1) and the power of the cosine
coefficients k = 1 ... 7 and k = 32 ... 79, with the Poisson error of the OBSERVED count of the subset.  For each design of random subsets
  layer 0  is exact, without draws: with the error from the EXPECTED count of the family prediction every coefficient is a mean over the subset of
           fixed numbers, one per L-function, and E[c_k^2] = (mean)^2 + variance of the subset mean for the design;
  layers 1 and 2 draw only the difference between the statistic as defined and the layer-0 statistic, on the same subsets (layer 1: 200 subsets;
           layer 2: further subsets where the error of the low-frequency power is above 2e-4, at most 1 600; pooled).
Designs, exactly as in the original scripts: random samples (compute_families.py, compute_moduli.py); samples with the composition of the family in
(sign, D mod M) for the 20 stratifications M = 1 (sign only), 3, ..., 840 (largest-remainder allocation, compute_moduli.py); m consecutive pieces
(pieces_repeat.py: lengths N//m and N//m + 1, the longer ones first, on a grid of blocks of the longer length, m blocks drawn without replacement;
layer 0 exact by conditioning on the blocks of ranks rem and rem + 1, which splits the subset into two independent simple random samples); one
piece (m = 1, every starting position, exact also for the observed count); the fixed consecutive bands and every starting position of a stretch
(no draws at all).  The line of Section 4.4 through the random samples and the consecutive bands is recomputed with the layered values and the
exact remaining shares W100 of reference_characters.json, and with the expectation over all starting positions of a stretch (W100 over all starts).
"""
import sys, os, json, math, time, numpy as np, mpmath as mp, scipy.special as sp, scipy.fft as FFT
from scipy.stats import chi2 as CHI2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
t0 = time.time(); ROOT, DATA, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT_DIR, exist_ok=True)
ONLY = sys.argv[sys.argv.index('--families') + 1].split(',') if '--families' in sys.argv else None   # Section 4.8: a family without original values
DT = 0.002; TG = np.round(np.arange(0, 8.0 + 1e-9, DT), 4); NT = len(TG); XF = 0.025; NF = 160; XB = 0.05; NB = 80; LB = 0.002; N = 6074
FAMS = ('B10', 'C100', 'D100') if not ONLY else tuple(ONLY); ALLM = [1, 3, 5, 7, 8, 9, 13, 16, 24, 40, 41, 120, 130, 131, 260, 360, 369, 520, 800, 840]; MS = [1, 3, 10, 30, 100, 300, 1000, 2000, N]
R1 = 200; TARGET = 2e-4; R2MAX = 1600; SEED = 34684
STATS = ('low', 'high', 'chi05', 'chi01')
# ---------------------------------------------------------------- the ratios conjecture on a grid in t (as compute_families.py and exact_layers.py)
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
print('grid', round(time.time() - t0), 's', flush=True)
TABS = {}
def kron(D, p):
    if p == 2: m = np.mod(D, 8); return np.where(D % 2 == 0, 0, np.where((m == 1) | (m == 7), 1, -1)).astype(np.float64)
    if p not in TABS: TABS[p] = np.array([0] + [1 if pow(r, (p - 1)//2, p) == 1 else -1 for r in range(1, p)], dtype=np.float64)
    return TABS[p][np.mod(D, p)]
P100 = [q for q in range(2, 101) if all(q % r for r in range(2, int(math.isqrt(q)) + 1))]; W100W = np.array([math.log(q)**2/q for q in P100])
def lgC(a, b):
    """log binomial(a, b) for arrays, -inf where b < 0 or b > a"""
    a = np.asarray(a, float); b = np.asarray(b, float); ok = (b >= 0) & (b <= a) & (a >= 0)
    return np.where(ok, sp.gammaln(np.maximum(a, 0) + 1) - sp.gammaln(np.maximum(b, 0) + 1) - sp.gammaln(np.maximum(a - b, 0) + 1), -np.inf)
# ---------------------------------------------------------------- original values (Sections 4.3 and 4.4)
ORS = json.load(open(os.path.join(DATA, 'reference_subsets.json'))); ORM = json.load(open(os.path.join(DATA, 'reference_moduli.json')))
ORP = json.load(open(os.path.join(DATA, 'reference_pieces_repeat.json'))); ORC = json.load(open(os.path.join(DATA, 'reference_characters.json')))
def ms(a): a = np.asarray(a, float); return [float(a.mean()), float(a.std(ddof=1)/math.sqrt(len(a))) if len(a) > 1 else 0.0, int(len(a))]
OUT = dict(study='III', section='4.7', subset_size=N, moduli=ALLM, pieces_m=MS, layer1_draws=R1, target_error=TARGET, layer2_max=R2MAX, seed=SEED, families={})
for f in FAMS:
    Ds, G = [], []
    for D, g in RZ.study_zeros(f, ROOT): Ds.append(D); G.append(g)
    Ds = np.array(Ds, dtype=np.int64); n = len(Ds); X = [theta(int(d), np.array(g))/math.pi for d, g in zip(Ds, G)]
    hm = np.stack([np.histogram(x[x < NB*XB], bins=np.arange(NB + 1)*XB)[0] for x in X]).astype(np.float32); del X, G
    L = np.log(np.abs(Ds).astype(float)/math.pi); Sg = (Ds < 0).astype(int); order = np.argsort(np.abs(Ds), kind='stable')
    Dv = np.empty((n, NB))
    for k in (0, 1):
        idx = np.nonzero(Sg == k)[0]; j = np.floor(L[idx]/LB).astype(np.int64); u = L[idx]/LB - j
        for jj in np.unique(j):
            q = idx[j == jj]; uu = u[j == jj][:, None]; Dv[q] = -((1 - uu)*node(k, int(jj))[None, :] + uu*node(k, int(jj) + 1)[None, :])
    Pbar = -Dv.mean(0); Dv += hm/XB                                                # Dv = density of one L-function minus its own prediction
    seP80 = np.sqrt(N*XB*Pbar)/(N*XB); U80 = FFT.dct(Dv/seP80[None, :], norm='ortho', axis=1)
    P40 = Pbar.reshape(NB//2, 2).mean(1); seP40 = np.sqrt(N*2*XB*P40)/(N*2*XB); U40 = FFT.dct(Dv.reshape(n, NB//2, 2).mean(2)/seP40[None, :], norm='ortho', axis=1)
    print('%-5s %d L-functions, per-L coefficients ready' % (f, n), round(time.time() - t0), 's', flush=True)
    def own(ii):
        m_ = len(ii); H = hm[ii].sum(0, dtype=np.float64); dm = Dv[ii].mean(0); c = FFT.dct(dm/(np.sqrt(np.maximum(H, 1))/(m_*XB)), norm='ortho')**2
        H40 = H.reshape(NB//2, 2).sum(1); dm40 = dm.reshape(NB//2, 2).mean(1); r40 = dm40/(np.sqrt(np.maximum(H40, 1))/(m_*2*XB))
        return np.array([c[1:8].sum()/80, c[32:80].sum()/80, c.sum()/80, float(np.mean(r40**2))])
    def lin_from(m80, m40, m_):
        c = (m_/N)*m80**2; c40 = (m_/N)*m40**2; return np.array([c[1:8].sum()/80, c[32:80].sum()/80, c.sum()/80, c40.sum()/40])
    def lin(ii): return lin_from(U80[ii].mean(0), U40[ii].mean(0), len(ii))
    def e0_from(E80, V80, E40, V40, m_):
        """layer 0 from the expectation and variance of the subset mean of every coefficient"""
        c = (m_/N)*(E80**2 + V80); c40 = (m_/N)*(E40**2 + V40); PARTS.append([float((m_/N)*(E80[1:8]**2).sum()/80), float((m_/N)*V80[1:8].sum()/80)])
        return np.array([c[1:8].sum()/80, c[32:80].sum()/80, c.sum()/80, c40.sum()/40])
    u80 = U80.mean(0); v80 = U80.var(0, ddof=1); u40 = U40.mean(0); v40 = U40.var(0, ddof=1)
    DES = {}; PARTS = []; W100S = {}
    DES['random'] = dict(E0=e0_from(u80, v80/N*(1 - N/n), u40, v40/N*(1 - N/n), N), samp=(lambda rng: rng.permutation(n)[:N]), size=N); DES['random']['parts'] = PARTS[-1]
    # ---- 20 stratifications by (sign, D mod M), largest-remainder allocation as compute_moduli.py
    for M in ALLM:
        tr = Sg*M + (Ds % M); uq, inv = np.unique(tr, return_inverse=True); Nc = np.bincount(inv); t_ = N*Nc/n; fl = np.floor(t_).astype(int)
        zb = N - fl.sum(); fl[np.argsort(-(t_ - fl), kind='stable')[:zb]] += 1; cl = [np.nonzero(inv == c)[0] for c in range(len(uq))]
        def cstat(UU):
            s1 = np.zeros((len(uq), UU.shape[1])); np.add.at(s1, inv, UU); s2 = np.zeros_like(s1); np.add.at(s2, inv, UU*UU); mc = s1/Nc[:, None]
            S2c = np.where(Nc[:, None] > 1, (s2 - Nc[:, None]*mc*mc)/np.maximum(Nc - 1, 1)[:, None], 0.0)
            return (fl[:, None]*mc).sum(0)/N, (fl[:, None]*(1 - fl[:, None]/Nc[:, None])*S2c).sum(0)/N**2
        E80, V80 = cstat(U80); E40, V40 = cstat(U40)
        if ONLY:
            Xc = np.stack([kron(Ds, p) for p in P100], 1); _, VX = cstat(Xc); VR = Xc.var(0, ddof=1)/N*(1 - N/n); W100S['mod%d' % M] = float(np.sum(W100W*VX)/np.sum(W100W*VR))
        DES['mod%d' % M] = dict(E0=e0_from(E80, V80, E40, V40, N), parts=None, size=N, classes=int(len(uq)),
                                samp=(lambda rng, cl=cl, fl=fl: np.concatenate([rng.choice(cl[c], fl[c], replace=False) for c in range(len(cl)) if fl[c] > 0]))); DES['mod%d' % M]['parts'] = PARTS[-1]
    print('%-5s stratifications: layer 0 low %s' % (f, ' '.join('%d:%.4f' % (M, DES['mod%d' % M]['E0'][0]) for M in ALLM)), round(time.time() - t0), 's', flush=True)
    # ---- m consecutive pieces (pieces_repeat.py): exact layer 0 by conditioning on the blocks of ranks rem and rem + 1
    PIE = {}
    for m in MS:
        if m == N: continue
        ln_ = N//m; rem = N - ln_*m
        if m == 1:
            Cs80 = np.vstack([np.zeros(NB), np.cumsum(U80[order], 0)]); Cs40 = np.vstack([np.zeros(NB//2), np.cumsum(U40[order], 0)]); ns = n - N
            w80 = (Cs80[N:N + ns] - Cs80[:ns])/N; w40 = (Cs40[N:N + ns] - Cs40[:ns])/N; del Cs80, Cs40
            E0 = e0_from(w80.mean(0), w80.var(0), w40.mean(0), w40.var(0), N); prt = PARTS[-1]; del w80, w40
            PIE[m] = dict(E0=E0, parts=prt, samp=(lambda rng: order[(lambda st: np.arange(st, st + N))(int(rng.integers(0, n - N)))]), size=N, starts=ns); continue
        Bk = ln_ + (1 if rem else 0); K = n//Bk; ob = order[:K*Bk].reshape(K, Bk)
        res = []
        for UU in (U80, U40):
            Ub = UU[ob]; Fb = Ub.sum(1); Ab = Ub[:, :ln_].sum(1); del Ub
            if rem == 0: Fb = Ab
            def srs(V, a, pre):
                """E and Var of the sum of a simple random sample of size a from the first (pre=True) or last blocks, for every position"""
                c1 = np.vstack([np.zeros(V.shape[1]), np.cumsum(V, 0)]); c2 = np.vstack([np.zeros(V.shape[1]), np.cumsum(V*V, 0)])
                idx = np.arange(K)
                if pre: s1 = c1[idx]; s2 = c2[idx]; pop = idx.astype(float)                       # population [0, p)
                else: s1 = c1[K] - c1[idx + 1]; s2 = c2[K] - c2[idx + 1]; pop = (K - 1 - idx).astype(float)   # population (q, K)
                with np.errstate(invalid='ignore', divide='ignore'):
                    mu = np.where(pop[:, None] > 0, s1/np.maximum(pop, 1)[:, None], 0.0); S2 = np.where(pop[:, None] > 1, (s2 - pop[:, None]*mu*mu)/np.maximum(pop - 1, 1)[:, None], 0.0)
                    fpc = np.where(pop > 0, a*(1 - a/np.maximum(pop, 1)), 0.0)
                return a*mu, fpc[:, None]*S2
            a_long, a_short = rem - 1, m - rem - 1
            EL, VL = srs(Fb, a_long, True); EL = EL + Fb; ES, VS = srs(Ab, a_short, False); ES = ES + Ab
            la = lgC(np.arange(K), a_long); lb = lgC(K - 1 - np.arange(K), a_short); lc = float(lgC(K, m))
            if rem == 0:
                EA, VA = Ab.mean(0)*m, m*(1 - m/K)*Ab.var(0, ddof=1); res.append((EA/N, VA/N**2, 1.0)); continue
            # P(rank rem in block p, rank rem + 1 in block q) = exp(la[p] + lb[q] - lc), p < q, is a product: prefix sums over p < q, rescaled by
            # la[q - 1] (la does not decrease), so every pair enters exactly and no number leaves the range of floating point
            d_ = UU.shape[1]; Z = np.hstack([np.ones((K, 1)), EL, VL + EL**2]); S = np.zeros((K, 1 + 2*d_)); s_ = np.zeros(1 + 2*d_); prev = None
            fin = np.isfinite(la)
            for q in range(1, K):
                if fin[q - 1]: s_ = Z[q - 1].copy() if prev is None else s_*math.exp(prev - la[q - 1]) + Z[q - 1]; prev = la[q - 1]
                S[q] = s_
            ok = fin[:-1] & np.isfinite(lb[1:]); w = np.zeros(K); w[1:][ok] = np.exp(lb[1:][ok] + la[:-1][ok] - lc); del Z
            S0 = S[:, :1]; S1 = S[:, 1:1 + d_]; S2 = S[:, 1 + d_:]; mass = float((w*S0[:, 0]).sum())
            totS = (w[:, None]*(S1 + S0*ES)).sum(0); tot = (w[:, None]*(S2 + 2*S1*ES + S0*(VS + ES**2))).sum(0); del S, S0, S1, S2
            res.append((totS/N, tot/N**2 - (totS/N)**2, mass))
        (E80, V80, mass80), (E40, V40, _m40) = res
        PIE[m] = dict(E0=e0_from(E80, V80, E40, V40, N), size=N, blocks=int(K), block=int(Bk), probability_mass=float(mass80), parts=PARTS[-1],
                      samp=(lambda rng, m=m, ln_=ln_, rem=rem, Bk=Bk: (lambda st: np.concatenate([order[s_:s_ + d_] for s_, d_ in zip(st, [ln_ + (1 if j < rem else 0) for j in range(m)])]))(np.sort(rng.choice(n//Bk, m, replace=False))*Bk)))
    for m, d in PIE.items():
        if m != 1: DES['m%d' % m] = d
    print('%-5s pieces: layer 0 low %s' % (f, ' '.join('%d:%.4f' % (m, d['E0'][0]) for m, d in PIE.items())), round(time.time() - t0), 's', flush=True)
    # ---- exact without draws: fixed bands, every starting position of one piece (m = 1, as pieces_repeat) and of a stretch (all n - N + 1 starts)
    nP = n//N; bands = np.array([own(order[j*N:(j + 1)*N]) for j in range(nP)]); bands_lin = np.array([lin(order[j*N:(j + 1)*N]) for j in range(nP)])
    Ch = np.vstack([np.zeros(NB), np.cumsum(hm[order].astype(np.float64), 0)]); Cd = np.vstack([np.zeros(NB), np.cumsum(Dv[order], 0)]); parts = []
    for a0 in range(0, n - N + 1, 50000):
        a1 = min(n - N + 1, a0 + 50000); Hw = Ch[a0 + N:a1 + N] - Ch[a0:a1]; Dw = (Cd[a0 + N:a1 + N] - Cd[a0:a1])/N
        cw = FFT.dct(Dw/(np.sqrt(np.maximum(Hw, 1))/(N*XB)), norm='ortho', axis=1)**2; H40 = Hw.reshape(-1, NB//2, 2).sum(2); D40 = Dw.reshape(-1, NB//2, 2).mean(2)
        r40 = D40/(np.sqrt(np.maximum(H40, 1))/(N*2*XB)); parts.append(np.stack([cw[:, 1:8].sum(1)/80, cw[:, 32:80].sum(1)/80, cw.sum(1)/80, (r40**2).mean(1)], 1))
    allst = np.concatenate(parts); del Ch, Cd, Hw, Dw, cw, H40, D40, r40, parts
    stretch = dict(starts=int(len(allst)), own=allst.mean(0).tolist(), own_sd=allst.std(0).tolist())
    m1 = dict(starts=int(n - N), own=allst[:n - N].mean(0).tolist(), layer0=PIE[1]['E0'].tolist(), layer0_low_mean_and_variance_terms=PIE[1]['parts'])
    # W100 over every starting position of a stretch (the remaining share of the weighted variance of the means of chi_D(p), p <= 100)
    vb = []; vb1 = []; th = []
    for p in P100:
        x = kron(Ds[order], p); mu = x.mean(); th.append(x.var()/N*(1 - N/n)); cx = np.concatenate([[0], np.cumsum(x)]); wm = (cx[N:] - cx[:-N])/N; vb.append(float(((wm - mu)**2).mean())); vb1.append(float(((wm[:n - N] - mu)**2).mean()))
    W100_all = float(np.sum(W100W*np.array(vb))/np.sum(W100W*np.array(th))); W100_m1 = float(np.sum(W100W*np.array(vb1))/np.sum(W100W*np.array(th))); del allst
    print('%-5s bands %d (mean low %.4f), stretch over %d starts: low %.4f high %.4f chi05 %.3f chi01 %.3f, W100 over all starts %.4f' % (f, nP, bands[:, 0].mean(), stretch['starts'], *stretch['own'], W100_all), round(time.time() - t0), 's', flush=True)
    # ---- layers 1 and 2 for every design with draws (common random numbers: draw r of every design uses the generator (SEED, r))
    R = {}
    for key, d in DES.items():
        O, Li = [], []
        for r in range(R1):
            ii = d['samp'](np.random.default_rng([SEED, r])); O.append(own(ii)); Li.append(lin(ii))
        dl = np.array(O) - np.array(Li); sd = dl.std(0, ddof=1); r2 = int(min(R2MAX, max(0, math.ceil((sd[0]/TARGET)**2) - R1)))
        for r in range(R1, R1 + r2):
            ii = d['samp'](np.random.default_rng([SEED, r])); O.append(own(ii)); Li.append(lin(ii))
        O = np.array(O); Li = np.array(Li); dl = O - Li; nd = len(dl); md = dl.mean(0); sdd = dl.std(0, ddof=1)/math.sqrt(nd)
        est = d['E0'] + md; plain = O.mean(0); plain_se = O.std(0, ddof=1)/math.sqrt(nd); chk = (Li.mean(0) - d['E0'])/(Li.std(0, ddof=1)/math.sqrt(nd))
        R[key] = dict(size=d['size'], draws=nd, layer0=dict(zip(STATS, d['E0'].tolist())), estimate={s_: [float(est[i]), float(sdd[i])] for i, s_ in enumerate(STATS)},
                      draws_only={s_: [float(plain[i]), float(plain_se[i])] for i, s_ in enumerate(STATS)}, error_ratio={s_: float(plain_se[i]/sdd[i]) for i, s_ in enumerate(STATS)},
                      layer0_check_z={s_: float(chk[i]) for i, s_ in enumerate(STATS)}, layer0_low_mean_and_variance_terms=d['parts'], **{k_: d[k_] for k_ in ('classes', 'blocks', 'block', 'probability_mass') if k_ in d})
        print('%-5s %-7s layer 0 %.4f  layered %.4f +- %.5f (%4d draws; draws only %.4f +- %.4f, %.1fx)  chi05 %.4f +- %.4f  layer-0 check z %+.2f' % (f, key, d['E0'][0], est[0], sdd[0], nd, plain[0], plain_se[0], plain_se[0]/sdd[0], est[2], sdd[2], chk[0]), round(time.time() - t0), 's', flush=True)
    # ---- original values
    orig43, orig44, origP = {}, {}, {}
    for typ in (('random', 'mod840', 'consecutive') if f in ORS['families'] else ()):
        q = [x for x in ORS['families'][f]['subsets'] if x['type'] == typ]; orig43[typ] = {'low': ms([x['low_k1_7'] for x in q]), 'high': ms([x['high_k32_79'] for x in q]), 'chi05': ms([x['chi2_005'] for x in q]), 'chi01': ms([x['chi2_01'] for x in q])}
    for typ, v in (ORM['families'][f]['types'].items() if f in ORM['families'] else ()):
        a = [q for ps_ in v['passes'].values() for q in ps_]; c = [q for ps_ in v['chi'].values() for q in ps_]; orig44[typ] = {'low': ms(a), 'chi05': ms(c)}
    if f in ORP['repetition']['families']:
        for m in MS:
            rp = ORP['repetition']['families'][f]['m'].get(str(m)); fr = ORP['first_run']['families'][f]['m'].get(str(m)) if f in ORP['first_run']['families'] else None
            origP[str(m)] = dict(repetition=None if rp is None else [rp['power_k1_7'], rp['power_k1_7_se'], rp['n_subsets']], first_run=None if fr is None else [fr['power_k1_7'], fr['power_k1_7_se'], fr['n_subsets']])
    OUT['families'][f] = dict(n=n, l=float(L.mean()), bands=dict(n=nP, own={s_: ms(bands[:, i]) for i, s_ in enumerate(STATS)}, layer0_statistic={s_: ms(bands_lin[:, i]) for i, s_ in enumerate(STATS)}, values_low=bands[:, 0].tolist()),
                              stretch=stretch, piece1=m1, W100_all_starts=W100_all, W100_piece1=W100_m1, designs=R, original={'4.3': orig43, '4.4': orig44, 'pieces': origP})
    if ONLY: OUT['families'][f]['W100_strata'] = W100S
    del hm, Dv, U80, U40
# ---------------------------------------------------------------- Section 4.4 recomputed: z against random samples and the line through two points
EX = ORC['exact'] if not ONLY else {f: {'types': {k: {'W100': v} for k, v in OUT['families'][f]['W100_strata'].items()}} for f in FAMS}; SUMC = ORC['summary']
def line(points):
    tot, nn, zz = 0.0, 0, []
    for f, (r_, s_, P) in points.items():
        q = np.array([(1 - t['W'])/(1 - s_['W']) for t in P]); pred = r_['v']*(1 - q) + s_['v']*q; d = np.array([t['v'] for t in P]) - pred
        C = np.diag([t['se']**2 for t in P]) + r_['se']**2*np.outer(1 - q, 1 - q) + s_['se']**2*np.outer(q, q); tot += float(d @ np.linalg.solve(C, d)); nn += len(P)
        zz += list(d/np.sqrt(np.diag(C)))
    return dict(chi2=tot, n=nn, p=float(CHI2.sf(tot, nn)), positive=int(sum(1 for z in zz if z > 0)), mean_z=float(np.mean(zz)), max_z=float(np.max(np.abs(zz))), z=[float(z) for z in zz])
REC = {}
for variant in (('original', 'layered_bands', 'layered_all_starts') if not ONLY else ('layered_all_starts',)):
    pts = {}
    for f in FAMS:
        F = OUT['families'][f]; Wt = EX[f]['types']; T = SUMC['families'][f]['types'] if f in SUMC['families'] else None
        if variant == 'original':
            r_ = dict(v=T['random']['low'], se=T['random']['se'], W=1.0); s_ = dict(v=T['consecutive']['low'], se=T['consecutive']['se'], W=Wt['consecutive']['W100'])
            P = [dict(v=T['mod%d' % M]['low'], se=T['mod%d' % M]['se'], W=Wt['mod%d' % M]['W100']) for M in ALLM]
        else:
            e = F['designs']['random']['estimate']['low']; r_ = dict(v=e[0], se=e[1], W=1.0)
            s_ = dict(v=F['bands']['own']['low'][0], se=F['bands']['own']['low'][1], W=Wt['consecutive']['W100']) if variant == 'layered_bands' else dict(v=F['stretch']['own'][0], se=0.0, W=F['W100_all_starts'])
            P = [dict(v=F['designs']['mod%d' % M]['estimate']['low'][0], se=F['designs']['mod%d' % M]['estimate']['low'][1], W=Wt['mod%d' % M]['W100']) for M in ALLM]
        pts[f] = (r_, s_, P)
    REC[variant] = line(pts)
    print('line through two points (%s): chi2 %.1f / %d (p %.2g), positive %d, mean z %+.2f, max |z| %.1f' % (variant, REC[variant]['chi2'], REC[variant]['n'], REC[variant]['p'], REC[variant]['positive'], REC[variant]['mean_z'], REC[variant]['max_z']), flush=True)
Z44 = {}
for f in FAMS:
    D = OUT['families'][f]['designs']; r_ = D['random']['estimate']['low']
    Z44[f] = {('mod%d' % M): float((D['mod%d' % M]['estimate']['low'][0] - r_[0])/math.hypot(D['mod%d' % M]['estimate']['low'][1], r_[1])) for M in ALLM}
    Z44[f]['bands'] = float((OUT['families'][f]['bands']['own']['low'][0] - r_[0])/math.hypot(OUT['families'][f]['bands']['own']['low'][1], r_[1]))
    Z44[f]['stretch_all_starts_ratio'] = OUT['families'][f]['stretch']['own'][0]/r_[0]
L0 = {}
for f in FAMS:
    F = OUT['families'][f]; Wt = EX[f]['types']; r0 = F['designs']['random']['layer0']['low']; s0 = F['piece1']['layer0'][0]; sW = F['W100_piece1']; rows = {}
    for M in ALLM:
        q = (1 - Wt['mod%d' % M]['W100'])/(1 - sW); pred = r0*(1 - q) + s0*q; ex = F['designs']['mod%d' % M]['layer0']['low']
        rows['mod%d' % M] = dict(exact=ex, line=pred, reduction_exact=(r0 - ex)/r0, reduction_line=(r0 - pred)/r0)
    L0[f] = dict(random=r0, stretch=s0, W100_stretch=sW, rows=rows, mean_ratio_of_reductions=float(np.mean([v['reduction_exact']/v['reduction_line'] for v in rows.values() if v['reduction_line'] > 0.01])))
    print('%-5s layer 0 exactly: reduction by the stratifications / reduction by the line (mean over those above 1 %%) %.2f; mod 840 %.3f against %.3f' % (f, L0[f]['mean_ratio_of_reductions'], rows['mod840']['reduction_exact'], rows['mod840']['reduction_line']), flush=True)
OUT['line_layer0_exact'] = L0
OUT['line'] = REC; OUT['z_44'] = Z44; OUT['W_k_weighted'] = {f: SUMC['families'][f]['W_k_weighted'] for f in FAMS if f in SUMC['families']}
for f in FAMS:
    print('%-5s z against random (layered): %s; stretch / random (exact over all starts) %.3f, W_k %.3f' % (f, ' '.join('%s %+.1f' % (k.replace('mod', ''), v) for k, v in Z44[f].items() if k.startswith('mod')), Z44[f]['stretch_all_starts_ratio'], OUT['W_k_weighted'].get(f, float('nan'))), flush=True)
json.dump(OUT, open(os.path.join(OUT_DIR, 'reference_recomputation%s.json' % ('_' + '_'.join(ONLY) if ONLY else '')), 'w')); print('done', round(time.time() - t0), 's', flush=True)
