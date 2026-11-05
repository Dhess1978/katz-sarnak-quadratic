#!/usr/bin/env python3
"""
Low-lying zeros of quadratic Dirichlet L-functions, study III: exact expectations and layered estimates of the low-frequency power of subsets,
and random samples balanced on the characters of all primes up to P (Section 4.3).

usage:  python3 -I exact_layers.py ZEROS_ROOT OUT_DIR          (about 40 minutes, 4 GB RAM; writes reference_layers.json)

The statistic is that of Sections 4.3-4.4: for a subset S of N L-functions the 80 deviations r_b = (density - mean prediction)/error in bins of 0.05,
with the Poisson error of the OBSERVED count of the subset (sqrt(H_b)), are decomposed by the orthonormal cosine transform; the power at k = 1 ... 7
is sum c_k^2 / 80.  Its expectation over all subsets of a given design is computed in layers:
  layer 0  exact, without any draws: with the error taken from the EXPECTED count of the family prediction (Pearson, as in study II) the coefficient
           c_k is a mean over the subset of a fixed number y_ik per L-function, so E[c_k^2] = (mean)^2 + variance of the subset mean, with the
           finite-population variance of the design (simple random sampling, sampling of pieces, stratification) - no random numbers;
  layer 1  draws only of the DIFFERENCE between the statistic as defined (observed count) and the layer-0 statistic, on the same subsets: it comes
           almost entirely from the two bins at x < 0.1, where a subset holds only 3-4 zeros and dividing by the square root of the observed count is
           strongly non-linear; the difference scatters several times less than the statistic itself;
  layer 2  further draws where layer 1 left the largest error (target error 2e-4), combined with layer 1 by the inverse of the variance.
Designs: random samples (pieces of length 1), pieces of l consecutive discriminants (sorted by |D|) on a grid at random positions, m = round(N/l)
pieces (the subset has m*l = N +- l/2 L-functions), one stretch (l = N; exact over all starting positions, also for the observed count), random
samples stratified by (sign, D mod 840).  Balanced samples: random samples of N whose means of chi_D(p) and chi_D(p)^2 equal those of the family for
every prime p <= P (random start, then random swaps that reduce the imbalance weighted by (ln p)^2/p, until every mean is within 0.05 of the error of a random sample or after 30 000 swaps);
100 balanced samples per P.  Their layer 0 follows without draws from the regression of y_ik on chi_D(p) and chi_D(p)^2 (p <= P, with a constant): a sample
balanced on these features keeps only the residual variance, E[c_k^2] = (mean)^2 + residual variance/N (1 - N/n); the regression is also given for larger P.
Coupling of the modes: the correlation of y_ik between modes k and k' and the variance of y_ik against the Poisson null model (zeros of every L-function
as a Poisson process with the density of the prediction; with the Pearson error its covariance is exactly N times the identity).
The same random numbers are used for all designs of a family (common random numbers).
"""
import sys, os, json, math, time, numpy as np, mpmath as mp, scipy.special as sp, scipy.fft as FFT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
t0 = time.time(); ROOT, OUT_DIR = sys.argv[1], sys.argv[2]; os.makedirs(OUT_DIR, exist_ok=True)
DT = 0.002; TG = np.round(np.arange(0, 8.0 + 1e-9, DT), 4); NT = len(TG); XF = 0.025; NF = 160; XB = 0.05; NB = 80; LB = 0.002; N = 6074
FAMS = ('B10', 'C100', 'D100'); LENGTHS = [1, 2, 3, 6, 10, 30, 100, 300, 1000]; PBAL = [7, 30, 100, 300]; PREG = [7, 30, 100, 300, 1000, 3000]; NBAL = 100; MAXSWAP = 30000; R1 = 200; TARGET = 2e-4; R2MAX = 1600; SEED = 3468
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
OUT = dict(study='III', subset_size=N, lengths=LENGTHS, balanced_P=PBAL, regression_P=PREG, layer1_draws=R1, target_error=TARGET, seed=SEED, families={})
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
    lam = N*XB*P.mean(0)
    def stat(ii):
        """statistic as defined (observed count) and the layer-0 statistic on the same subset"""
        m_ = len(ii); H = hm[ii].sum(0); dm = Dv[ii].mean(0); se = np.sqrt(np.maximum(H, 1))/(m_*XB)
        own = float(((A @ (dm/se))**2).sum()/80); lin = float((m_/N)*((Y[ii].mean(0))**2).sum()/80); return own, lin
    def pea(ii):
        m_ = len(ii); dm = Dv[ii].mean(0); seq = np.sqrt(m_*XB*P[ii].mean(0))/(m_*XB); return float(((A @ (dm/seq))**2).sum()/80)
    D_ = {}
    # ---- designs: exact layer 0 and a sampler
    ybar = Y.mean(0); S2 = Y.var(0, ddof=1)
    Cm = np.corrcoef(Y.T); coupling = dict(correlation=Cm.round(6).tolist(), variance_over_poisson=(S2/N).tolist(), pairs={'%d-%d' % (a + 1, b + 1): float(Cm[a, b]) for a, b in ((0, 6), (1, 5), (2, 4), (0, 2), (1, 3))})
    print('%-5s coupling of modes: corr(1,7) %.3f (2,6) %.3f (3,5) %.3f; variance / Poisson %s' % (f, Cm[0, 6], Cm[1, 5], Cm[2, 4], np.round(S2/N, 3).tolist()), round(time.time() - t0), 's', flush=True)
    def regression(Pm):
        """residual variance of y_ik after the regression on 1, chi_D(p), chi_D(p)^2 for p <= P (normal equations in blocks of rows)"""
        pr_ = primes(Pm); q = 1 + 2*len(pr_); G = np.zeros((q, q)); bY = np.zeros((q, 7))
        for a0 in range(0, n, 20000):
            Dc = Ds[a0:a0 + 20000]; ch = np.stack([kron(Dc, p_) for p_ in pr_], 1); Fc = np.hstack([np.ones((len(Dc), 1)), ch, ch*ch]); G += Fc.T @ Fc; bY += Fc.T @ Y[a0:a0 + 20000]
        coef = np.linalg.lstsq(G, bY, rcond=None)[0]; rss = (Y*Y).sum(0) - (coef*bY).sum(0); S2r = np.maximum(rss, 0)/(n - 1)
        return dict(primes=len(pr_), layer0=float((ybar**2 + S2r/N*(1 - N/n)).sum()/80), explained_share=(1 - S2r/S2).tolist(), residual_variance_over_poisson=(S2r/N).tolist())
    REG = {}
    for Pm in PREG + ([10000] if f == 'B10' else []):
        REG[str(Pm)] = regression(Pm)
        print('%-5s regression on p <= %d: layer 0 %.5f  explained share by mode %s' % (f, Pm, REG[str(Pm)]['layer0'], np.round(REG[str(Pm)]['explained_share'], 3).tolist()), round(time.time() - t0), 's', flush=True)
    for l in LENGTHS:
        if l == 1:
            E0 = float((ybar**2 + S2/N*(1 - N/n)).sum()/80); samp = (lambda rng: rng.choice(n, N, replace=False)); Ns = N
        else:
            M_ = n//l; Z = Y[order[:M_*l]].reshape(M_, l, 7).mean(1); m_ = int(round(N/l)); Ns = m_*l
            E0 = float((Ns/N)*(Z.mean(0)**2 + Z.var(0, ddof=1)/m_*(1 - m_/M_)).sum()/80)
            samp = (lambda rng, l=l, M_=M_, m_=m_: order[(rng.choice(M_, m_, replace=False)[:, None]*l + np.arange(l)[None, :]).ravel()])
        D_['l%d' % l] = dict(E0=E0, samp=samp, size=Ns)
    cls, inv = np.unique(S*840 + Ds % 840, return_inverse=True); cnt = np.bincount(inv); q_ = N*cnt/n; kc = np.floor(q_).astype(np.int64)
    kc[np.argsort(-(q_ - kc), kind='stable')[:N - int(kc.sum())]] += 1; members = [np.nonzero(inv == c)[0] for c in range(len(cls))]
    mc_ = np.array([Y[mm].mean(0) for mm in members]); S2c = np.array([Y[mm].var(0, ddof=1) if len(mm) > 1 else np.zeros(7) for mm in members])
    E0s = float((((kc[:, None]*mc_).sum(0)/N)**2 + (kc[:, None]*(1 - kc[:, None]/cnt[:, None])*S2c).sum(0)/N**2).sum()/80)
    D_['mod840'] = dict(E0=E0s, samp=(lambda rng: np.concatenate([rng.choice(mm, k_, replace=False) for mm, k_ in zip(members, kc) if k_ > 0])), size=N)
    # ---- one stretch: exact over all starting positions, for the statistic as defined as well
    Ch = np.vstack([np.zeros(NB), np.cumsum(hm[order], 0)]); Cd = np.vstack([np.zeros(NB), np.cumsum(Dv[order], 0)]); Cy = np.vstack([np.zeros(7), np.cumsum(Y[order], 0)])
    Hw = Ch[N:] - Ch[:-N]; Dw = (Cd[N:] - Cd[:-N])/N; Ww = (Dw/(np.sqrt(np.maximum(Hw, 1))/(N*XB))) @ A.T
    stretch = dict(own=float((Ww**2).sum(1).mean()/80), layer0=float((((Cy[N:] - Cy[:-N])/N)**2).sum(1).mean()/80), starts=int(len(Ww)), own_k=((Ww**2).mean(0)/80).tolist(),
                   layer0_k=((((Cy[N:] - Cy[:-N])/N)**2).mean(0)/80).tolist())
    del Ch, Cd, Cy, Hw, Dw, Ww
    # ---- layers 1 and 2 for every design (common random numbers: draw r of every design uses the generator (SEED, r))
    R = {}
    for key, d in D_.items():
        own, lin = [], []
        for r in range(R1):
            o_, l_ = stat(d['samp'](np.random.default_rng([SEED, r]))); own.append(o_); lin.append(l_)
        dl = np.array(own) - np.array(lin); m1, s1 = float(dl.mean()), float(dl.std(ddof=1)/math.sqrt(R1)); sd = float(dl.std(ddof=1))
        r2 = int(min(R2MAX, max(0, math.ceil((sd/TARGET)**2) - R1))); o2, l2 = [], []
        for r in range(R1, R1 + r2):
            o_, l_ = stat(d['samp'](np.random.default_rng([SEED, r]))); o2.append(o_); l2.append(l_)
        if r2 > 1:
            d2 = np.array(o2) - np.array(l2); m2, s2 = float(d2.mean()), float(d2.std(ddof=1)/math.sqrt(r2)); w1, w2 = 1/s1**2, 1/s2**2; mc, sc = (m1*w1 + m2*w2)/(w1 + w2), 1/math.sqrt(w1 + w2)
        else: m2 = s2 = None; mc, sc = m1, s1
        own_all = np.array(own + o2); plain = (float(own_all.mean()), float(own_all.std(ddof=1)/math.sqrt(len(own_all))))
        R[key] = dict(size=d['size'], layer0=d['E0'], layer1=[d['E0'] + m1, s1], layer2=None if m2 is None else [d['E0'] + m2, s2], estimate=[d['E0'] + mc, sc],
                      difference=[mc, sc], sd_statistic=float(np.array(own).std(ddof=1)), sd_difference=sd, draws=[R1, r2], draws_only=list(plain))
        print('%-5s %-7s layer 0 %.4f  layer 1 %.4f +- %.4f  layer 2 (%4d) %s  ->  %.4f +- %.4f   (draws only %.4f +- %.4f; sd %.4f / %.4f)'
              % (f, key, d['E0'], d['E0'] + m1, s1, r2, '-' if m2 is None else '%.4f +- %.4f' % (d['E0'] + m2, s2), d['E0'] + mc, sc, plain[0], plain[1], R[key]['sd_statistic'], sd), round(time.time() - t0), 's', flush=True)
    # ---- Pearson as a check: the same statistic with the expected count of the subset prediction (as study II): layer 0 needs no correction
    pr_chk = {}
    lo2 = np.zeros(NB, bool); lo2[:2] = True
    for key in ('l1', 'l30', 'l1000'):
        rows = []
        for r in range(R1):
            ii = D_[key]['samp'](np.random.default_rng([SEED, r])); m_ = len(ii); H = hm[ii].sum(0); dm = Dv[ii].mean(0); se_o = np.sqrt(np.maximum(H, 1))/(m_*XB); se_p = np.sqrt(m_*XB*P[ii].mean(0))/(m_*XB)
            se_m = np.where(lo2, se_p, se_o); o_, l_ = stat(ii); rows.append([o_, float(((A @ (dm/se_p))**2).sum()/80), float(((A @ (dm/se_m))**2).sum()/80), l_])
        rows = np.array(rows); pr_chk[key] = dict(observed_minus_pearson=[float((rows[:, 0] - rows[:, 1]).mean()), float((rows[:, 0] - rows[:, 1]).std(ddof=1)/math.sqrt(R1))],
            observed_with_pearson_in_two_lowest_bins_minus_pearson=[float((rows[:, 2] - rows[:, 1]).mean()), float((rows[:, 2] - rows[:, 1]).std(ddof=1)/math.sqrt(R1))],
            pearson_minus_layer0=[float((rows[:, 1] - rows[:, 3]).mean()), float((rows[:, 1] - rows[:, 3]).std(ddof=1)/math.sqrt(R1))])
    # ---- balanced samples: means of chi_D(p) and chi_D(p)^2 equal to the family for every prime p <= P
    BAL = {}
    for Pm in PBAL + ([1000] if f == 'B10' else []):
        pr_ = primes(Pm); F = np.stack([kron(Ds, q) for q in pr_] + [kron(Ds, q)**2 for q in pr_], 1).astype(np.float32); w = np.array([math.log(q)**2/q for q in pr_]*2, np.float32)
        mu = F.mean(0, dtype=np.float64).astype(np.float32); sdF = np.maximum(F.std(0, dtype=np.float64), 1e-12).astype(np.float32); vals, lins, imb, its = [], [], [], []
        step = (1/(sdF*math.sqrt(N))).astype(np.float32)
        for r in range(NBAL):
            rng = np.random.default_rng([SEED, 10**6 + 1000*Pm + r]); perm = rng.permutation(n); mem = perm[:N].copy(); pool = perm[N:N + min(n - N, 50000)].copy(); Fp = F[pool].copy()
            z = ((F[mem].sum(0, dtype=np.float64)/N - mu)/sdF*math.sqrt(N)).astype(np.float32); q0 = float(np.sum(w*z*z)); it = 0
            for it in range(MAXSWAP):
                pos = int(rng.integers(0, N)); a = mem[pos]; pc = rng.integers(0, len(pool), 256)
                zn = z[None, :] + (Fp[pc] - F[a][None, :])*step[None, :]; qn = (w[None, :]*zn*zn).sum(1); b = int(np.argmin(qn))
                if qn[b] < q0:
                    c = pool[pc[b]]; mem[pos] = c; pool[pc[b]] = a; Fp[pc[b]] = F[a]; z = zn[b]; q0 = float(qn[b])
                if it % 200 == 0 and np.max(np.abs(z)) < 0.05: break
            o_, l_ = stat(np.sort(mem)); vals.append(o_); lins.append(l_); imb.append(float(np.max(np.abs(z)))); its.append(it)
        vals = np.array(vals); lins = np.array(lins); dif = vals - lins; e0 = REG[str(Pm)]['layer0'] if str(Pm) in REG else regression(Pm)['layer0']; mdf, sdf = float(dif.mean()), float(dif.std(ddof=1)/math.sqrt(NBAL))
        BAL[str(Pm)] = dict(primes=len(pr_), mean=float(vals.mean()), se=float(vals.std(ddof=1)/math.sqrt(NBAL)), subsets=NBAL, layer0_statistic_mean=float(lins.mean()), layer0_statistic_se=float(lins.std(ddof=1)/math.sqrt(NBAL)),
                            layer0_regression=e0, difference=[mdf, sdf], estimate=[e0 + mdf, sdf], max_imbalance_in_random_sd=float(np.max(imb)), median_imbalance=float(np.median(imb)), swaps_median=int(np.median(its)))
        print('%-5s balanced on p <= %d (%d primes): %.4f +- %.4f; layer-0 statistic %.4f +- %.4f against regression %.4f; layered %.4f +- %.4f  (imbalance median %.3f, max %.3f random sd; median %d swaps)'
              % (f, Pm, len(pr_), vals.mean(), vals.std(ddof=1)/math.sqrt(NBAL), lins.mean(), lins.std(ddof=1)/math.sqrt(NBAL), e0, e0 + mdf, sdf, np.median(imb), np.max(imb), np.median(its)), round(time.time() - t0), 's', flush=True)
        del F
    OUT['families'][f] = dict(n=n, l=float(L.mean()), expected_zeros_lowest_bins=[float(lam[0]), float(lam[1])], random_k_layer0=((ybar**2 + S2/N*(1 - N/n))/80).tolist(), designs=R, stretch=stretch, pearson_check=pr_chk, balanced=BAL, regression=REG, coupling=coupling)
    print('%-5s stretch exact over %d starts: statistic %.4f (layer 0 %.4f); Pearson check %s' % (f, stretch['starts'], stretch['own'], stretch['layer0'], pr_chk), round(time.time() - t0), 's', flush=True)
json.dump(OUT, open(os.path.join(OUT_DIR, 'reference_layers.json'), 'w')); print('done', round(time.time() - t0), 's', flush=True)
