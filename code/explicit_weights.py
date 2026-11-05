#!/usr/bin/env python3
"""
Low-lying zeros of quadratic Dirichlet L-functions, study III, Section 4.7: the weights of the explicit formula against the regression of the
contributions y_ik on the characters, the dual term of the ratios conjecture for the subfamilies with fixed chi_D(p), and the low-frequency power
of stratifications and of one stretch predicted from the characters alone.

usage:  python3 -I explicit_weights.py ZEROS_ROOT OUT_DIR          (about 20 minutes on two cores, 4 GB RAM; writes reference_explicit_weights.json)
        python3 -I explicit_weights.py ZEROS_ROOT OUT_DIR --families E100      (Section 4.8; writes reference_explicit_weights_E100.json)

For one L-function the explicit formula gives the number of zeros 0 < gamma < T exactly; the part that depends on the character is
    -(1/pi) sum_n Lambda(n) chi_D(n) n^(-1/2) int cos(t ln n) dt,
so the deviation of the density in a bin [x_b, x_b+1] of the unfolded variable x = theta_D(t)/pi (edges t_b from theta_D) has, for every prime p,
the coefficient  beta1_pb = -(1/(pi XB)) sum_{k odd} ln p p^(-k/2) I(k ln p; b)  on chi_D(p)  and  beta2_pb (k even >= 2) on chi_D(p)^2,
I(w; b) = (sin w t_b+1 - sin w t_b)/w.  The contribution of one L-function to the cosine mode k (error from the expected count, Section 4.6) is
y_ik = sum_b A_kb (density - prediction)_b / se_b, hence the predicted weights gamma_pk = sum_b A_kb beta_pb / se_b, without any fit.
These diagonal terms (n = p^k) are not the whole dependence on chi_D(p): in the ratios conjecture the subfamily with chi_D(p) = e (e = +1, -1, 0)
has the local factor F_e(a, g) = (1 - e p^(-1/2-g))/(1 - e p^(-1/2-a)) at p instead of the family mean F_p = (p^2 - z)/((p + 1)(p - z)), z = p^(2it),
in both terms; the diagonal term gives back beta1, beta2, the dual term -(1/pi) Re[(|D|/pi)^(-it) X ... zeta(1 - 2it) A(-it, it)] (the term that gives
the sine of W_USp) changes by the factor A_e/A_p, A_e = F_e(-it, it) (p - z)/(p - 1), A_p = 1 - (z - 1)/(p^2 - 1).  Its coefficients on chi_D(p)
((d_+ - d_-)/2) and chi_D(p)^2 ((d_+ + d_-)/2 - d_0) are added as the dual weights; every prediction is given with the diagonal weights alone and
with diagonal + dual weights.
(1) the regression of y_ik on 1, chi_D(p), chi_D(p)^2 for p <= 100 against gamma_pk (mean over the family; the coefficients vary with l = ln(|D|/pi));
(2) the share of the variance of y_ik explained by the prediction y^_ik = sum_p gamma_pk(l_i) (chi_D(p) - mean) + ... (p <= 1000), without a fit;
(3) every stratification of Section 4.4: layer 0 predicted from the characters alone, E0(random) - [E0_random(y^) - E0_stratified(y^)], against the
    exact layer 0 from the zeros;
(4) one stretch, every starting position: the same prediction, exactly for p <= 1000 and with the sum over primes up to 2*10^5, prime by prime
    (variance of the means of chi_D(p) and chi_D(p)^2 over all windows, cross terms between different primes neglected - checked for p <= 1000).
"""
import sys, os, json, math, time, numpy as np, mpmath as mp, scipy.special as sp, scipy.fft as FFT
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
t0 = time.time(); ROOT, OUT_DIR = sys.argv[1], sys.argv[2]; os.makedirs(OUT_DIR, exist_ok=True)
DT = 0.002; TG = np.round(np.arange(0, 8.0 + 1e-9, DT), 4); NT = len(TG); XF = 0.025; NF = 160; XB = 0.05; NB = 80; LB = 0.002; N = 6074; LN = 0.01
ONLY = sys.argv[sys.argv.index('--families') + 1].split(',') if '--families' in sys.argv else None
FAMS = ('B10', 'C100', 'D100') if not ONLY else tuple(ONLY); ALLM = [1, 3, 5, 7, 8, 9, 13, 16, 24, 40, 41, 120, 130, 131, 260, 360, 369, 520, 800, 840]; PREGR = 100; PHAT = 1000; PDIAG = 200000
# ---------------------------------------------------------------- the ratios conjecture on a grid in t (as exact_layers.py)
Mp = 200000; s = np.ones(Mp + 1, bool); s[:2] = False
for p in range(2, int(Mp**0.5) + 1):
    if s[p]: s[p*p::p] = False
PRM = np.nonzero(s)[0].astype(float); lp = np.log(PRM); PRI = np.nonzero(s)[0]
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
def dual_at(l, kap, primes, tp):
    """dual term of the ratios conjecture for the subfamilies chi_D(p) = +1, -1, 0: the integral over [0, t] of its coefficients on chi_D(p) and
    chi_D(p)^2 in the density of zeros, at the points tp (primes x len(tp)); computed in chunks of primes on the grid TG"""
    base = np.exp(-1j*TG*l)*Z2[kap]; tp = np.asarray(tp, float); j = np.clip(np.floor(tp/DT + 1e-9).astype(int), 0, NT - 2); w = (tp - TG[j])/DT
    ob = np.empty((len(primes), len(tp))); oc = np.empty_like(ob)
    for a0 in range(0, len(primes), 300):
        pp = np.asarray(primes[a0:a0 + 300], float)[:, None]; e = np.exp(1j*TG[None, :]*np.log(pp)); z = e*e
        Ap = (pp*pp - z)/(pp*pp - 1); Yi = (pp - z)/(pp - 1); u = pp**-0.5/e; v = pp**-0.5*e; dd = []
        for A in ((1 - u)/(1 - v)*Yi, (1 + u)/(1 + v)*Yi, Yi):
            d = -np.real(base[None, :]*(A/Ap - 1))/math.pi; d[:, 0] = 3*d[:, 1] - 3*d[:, 2] + d[:, 3]; dd.append(d)
        for O, d in ((ob, 0.5*(dd[0] - dd[1])), (oc, 0.5*(dd[0] + dd[1]) - dd[2])):
            C = np.zeros_like(d); C[:, 1:] = np.cumsum(0.5*(d[:, 1:] + d[:, :-1])*DT, 1); O[a0:a0 + 300] = C[:, j]*(1 - w) + C[:, j + 1]*w
    return ob, oc
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
A = FFT.dct(np.eye(NB), norm='ortho', axis=0)[1:8]
TABS = {}
def kron(D, p):
    """Kronecker symbol (D/p) for a prime p: (D/2) from D mod 8, for odd p the table of quadratic residues mod p"""
    if p == 2: m = np.mod(D, 8); return np.where(D % 2 == 0, 0, np.where((m == 1) | (m == 7), 1, -1)).astype(np.float64)
    tb = TABS.get(p)
    if tb is None:
        r = np.arange(1, p, dtype=np.int64); tb = -np.ones(p); tb[(r*r) % p] = 1.0; tb[0] = 0.0
        if p < 20000: TABS[p] = tb
    return tb[np.mod(D, p)]
def weights(l, kap, primes, seP):
    """predicted gamma1, gamma2 (primes x 7) for one value of l and sign: bin edges in t from theta, sum over the prime powers (diagonal),
    and the same with the dual term added"""
    th = (0.5*TG*l + IMLG[kap])/math.pi; te = np.interp(np.arange(NB + 1)*XB, th, TG)
    lp_ = np.log(primes.astype(float)); b1 = np.zeros((len(primes), NB)); b2 = np.zeros((len(primes), NB)); k = 1
    while True:
        amp = lp_*primes.astype(float)**(-k/2); sel = amp > 1e-13
        if not sel.any(): break
        w = k*lp_[sel]; I = (np.sin(w[:, None]*te[None, 1:]) - np.sin(w[:, None]*te[None, :-1]))/w[:, None]
        (b1 if k % 2 else b2)[sel] += amp[sel, None]*I; k += 1
    b1 *= -1/(math.pi*XB); b2 *= -1/(math.pi*XB); d1, d2 = dual_at(l, kap, primes, te); d1 = np.diff(d1, axis=1)/XB; d2 = np.diff(d2, axis=1)/XB
    return (b1/seP[None, :]) @ A.T, (b2/seP[None, :]) @ A.T, ((b1 + d1)/seP[None, :]) @ A.T, ((b2 + d2)/seP[None, :]) @ A.T
XW = [(0.1, 0.4), (0.4, 0.8), (0.8, 1.5), (1.5, 2.5), (2.5, 4.0)]; PW = [2, 3, 5, 7, 11, 13]
def window_weights(l, kap, primes):
    """explicit-formula coefficients of chi_D(p) and chi_D(p)^2 on the number of zeros in the windows XW of the unfolded variable"""
    th = (0.5*TG*l + IMLG[kap])/math.pi; te = np.interp(np.array(XW, float).ravel(), th, TG).reshape(len(XW), 2); out1 = np.zeros((len(primes), len(XW))); out2 = np.zeros_like(out1)
    for i, p in enumerate(primes):
        lp_ = math.log(p); k = 1
        while lp_*p**(-k/2) > 1e-13:
            w = k*lp_; I = (np.sin(w*te[:, 1]) - np.sin(w*te[:, 0]))/w; (out1 if k % 2 else out2)[i] += lp_*p**(-k/2)*I; k += 1
    d1, d2 = dual_at(l, kap, primes, te.ravel()); d1 = d1[:, 1::2] - d1[:, 0::2]; d2 = d2[:, 1::2] - d2[:, 0::2]
    return -out1/math.pi, -out2/math.pi, -out1/math.pi + d1, -out2/math.pi + d2
OUT = dict(study='III', section='4.7', windows_x=XW, window_primes=PW, subset_size=N, regression_P=PREGR, prediction_P=PHAT, diagonal_P=PDIAG, l_node=LN, families={})
for f in FAMS:
    Ds, G = [], []
    for D, g in RZ.study_zeros(f, ROOT): Ds.append(D); G.append(g)
    Ds = np.array(Ds, dtype=np.int64); n = len(Ds); X = [theta(int(d), np.array(g))/math.pi for d, g in zip(Ds, G)]
    hm = np.stack([np.histogram(x[x < NB*XB], bins=np.arange(NB + 1)*XB)[0] for x in X]).astype(np.float64)
    L = np.log(np.abs(Ds).astype(float)/math.pi); Sg = (Ds < 0).astype(int); order = np.argsort(np.abs(Ds), kind='stable')
    # ---- (0) windows of the unfolded variable: regression of the number of zeros on chi_D(p), chi_D(p)^2 (with l and the sign) against the explicit formula
    CW = np.stack([np.array([np.sum((x > a) & (x <= b)) for (a, b) in XW], float) for x in X]); del X, G
    PWa = np.array(PW); wl = np.zeros((n, len(PW), len(XW))); wl2 = np.zeros_like(wl); wf = np.zeros_like(wl); wf2 = np.zeros_like(wl); jn0 = np.round(L/LN).astype(np.int64)
    for kap in (0, 1):
        for jj in np.unique(jn0[Sg == kap]):
            q = np.nonzero((Sg == kap) & (jn0 == jj))[0]; a1, a2, f1, f2 = window_weights(jj*LN, kap, PWa); wl[q] = a1[None]; wl2[q] = a2[None]; wf[q] = f1[None]; wf2[q] = f2[None]
    WIN = {}
    for i, p in enumerate(PW):
        x = kron(Ds, p); Fw = np.stack([np.ones(n), x, x*x, L, Sg.astype(float)], 1); cf, *_ = np.linalg.lstsq(Fw, CW, rcond=None); rs = CW - Fw @ cf
        sew = np.sqrt(np.outer(np.diag(np.linalg.inv(Fw.T @ Fw)), (rs*rs).sum(0)/(n - 5)))
        WIN[str(p)] = dict(fitted_chi=cf[1].tolist(), fitted_chi_se=sew[1].tolist(), predicted_chi=wl[:, i].mean(0).tolist(), predicted_chi_with_dual=wf[:, i].mean(0).tolist(),
                           fitted_chi2=cf[2].tolist(), fitted_chi2_se=sew[2].tolist(), predicted_chi2=wl2[:, i].mean(0).tolist(), predicted_chi2_with_dual=wf2[:, i].mean(0).tolist())
    zd = np.array([[(a - b)/c for a, b, c in zip(W_['fitted_chi'], W_['predicted_chi'], W_['fitted_chi_se'])] for W_ in WIN.values()])
    zf = np.array([[(a - b)/c for a, b, c in zip(W_['fitted_chi'], W_['predicted_chi_with_dual'], W_['fitted_chi_se'])] for W_ in WIN.values()])
    zf2 = np.array([[(a - b)/c for a, b, c in zip(W_['fitted_chi2'], W_['predicted_chi2_with_dual'], W_['fitted_chi2_se'])] for W_ in WIN.values()])
    WINS = dict(chi2_diagonal=float((zd**2).sum()), chi2_with_dual=float((zf**2).sum()), chi2_chi2_with_dual=float((zf2**2).sum()), n=int(zd.size), max_abs_z_diagonal=float(np.abs(zd).max()), max_abs_z_with_dual=float(np.abs(zf).max()))
    print('%-5s windows x %s, primes %s: chi^2 of fitted - predicted (chi_D(p)) diagonal %.1f, with the dual term %.1f (chi_D(p)^2 %.1f) for %d coefficients; max |z| %.1f / %.1f' % (
          f, XW, PW, WINS['chi2_diagonal'], WINS['chi2_with_dual'], WINS['chi2_chi2_with_dual'], WINS['n'], WINS['max_abs_z_diagonal'], WINS['max_abs_z_with_dual']), round(time.time() - t0), 's', flush=True)
    del wl, wl2, wf, wf2, CW
    P = np.empty((n, NB))
    for k in (0, 1):
        idx = np.nonzero(Sg == k)[0]; j = np.floor(L[idx]/LB).astype(np.int64); u = L[idx]/LB - j
        for jj in np.unique(j):
            q = idx[j == jj]; uu = u[j == jj][:, None]; P[q] = (1 - uu)*node(k, int(jj))[None, :] + uu*node(k, int(jj) + 1)[None, :]
    Dv = hm/XB - P; seP = np.sqrt(N*XB*P.mean(0))/(N*XB); Y = (Dv/seP[None, :]) @ A.T; del P, hm, Dv
    ybar = Y.mean(0); S2 = Y.var(0, ddof=1); E0r = (ybar**2 + S2/N*(1 - N/n))
    # ---- predicted weights on a grid of l (step 0.01) for both signs, linear interpolation for every L-function
    pr_hat = PRI[PRI <= PHAT]; pr_reg = PRI[PRI <= PREGR]; jn = np.floor(L/LN).astype(np.int64); un = L/LN - jn; j0 = int(jn.min()); J = int(jn.max()) - j0 + 2
    TB1 = np.zeros((2, J, len(pr_hat), 7)); TB2 = np.zeros_like(TB1); TF1 = np.zeros_like(TB1); TF2 = np.zeros_like(TB1); nn_ = 0
    for kap in (0, 1):
        need = set()
        for jj in np.unique(jn[Sg == kap]): need |= {int(jj), int(jj) + 1}
        for j2 in need: TB1[kap, j2 - j0], TB2[kap, j2 - j0], TF1[kap, j2 - j0], TF2[kap, j2 - j0] = weights(j2*LN, kap, pr_hat, seP); nn_ += 1
    lo_ = jn - j0; w_hi = un[:, None]; w_lo = 1 - w_hi
    def gam(TB, i): return w_lo*TB[Sg, lo_, i] + w_hi*TB[Sg, lo_ + 1, i]          # weights of prime number i for every L-function (n x 7)
    print('%-5s %d L-functions, weights on %d nodes of l' % (f, n, nn_), round(time.time() - t0), 's', flush=True)
    # ---- (1) regression on p <= 100 against the predicted weights
    nr = len(pr_reg); F_ = np.empty((n, 1 + 2*nr))
    F_[:, 0] = 1.0
    for i, p in enumerate(pr_reg): x = kron(Ds, int(p)); F_[:, 1 + i] = x; F_[:, 1 + nr + i] = x*x
    coef, *_ = np.linalg.lstsq(F_, Y, rcond=None); res = Y - F_ @ coef; s2r = (res*res).sum(0)/(n - F_.shape[1])
    cov = np.linalg.inv(F_.T @ F_); se = np.sqrt(np.outer(np.diag(cov), s2r))
    gm1 = np.array([gam(TB1, i).mean(0) for i in range(nr)]); gm2 = np.array([gam(TB2, i).mean(0) for i in range(nr)])
    gf1 = np.array([gam(TF1, i).mean(0) for i in range(nr)]); gf2 = np.array([gam(TF2, i).mean(0) for i in range(nr)])
    fit1, fit2, se1, se2 = coef[1:1 + nr], coef[1 + nr:], se[1:1 + nr], se[1 + nr:]
    zz1 = (fit1 - gm1)/se1; zz2 = (fit2 - gm2)/se2; zf1 = (fit1 - gf1)/se1; zf2 = (fit2 - gf2)/se2
    def slope(fv, gv, sv): w = 1/sv**2; return float((w*fv*gv).sum()/(w*gv*gv).sum()), float(1/math.sqrt((w*gv*gv).sum()))
    slope_k = [slope(fit1[:, k], gm1[:, k], se1[:, k]) for k in range(7)]; slope_fk = [slope(fit1[:, k], gf1[:, k], se1[:, k]) for k in range(7)]
    reg = dict(slope_chi_by_mode=slope_k, primes=pr_reg.tolist(), fitted_chi=fit1.tolist(), fitted_chi_se=se1.tolist(), predicted_chi=gm1.tolist(), fitted_chi2=fit2.tolist(), fitted_chi2_se=se2.tolist(), predicted_chi2=gm2.tolist(),
               slope_chi=slope(fit1, gm1, se1), slope_chi2=slope(fit2, gm2, se2), chi2_chi=float((zz1**2).sum()), chi2_chi2=float((zz2**2).sum()), n_coef=int(zz1.size),
               predicted_chi_with_dual=gf1.tolist(), predicted_chi2_with_dual=gf2.tolist(), slope_chi_with_dual=slope(fit1, gf1, se1), slope_chi2_with_dual=slope(fit2, gf2, se2),
               slope_chi_with_dual_by_mode=slope_fk, chi2_chi_with_dual=float((zf1**2).sum()), chi2_chi2_with_dual=float((zf2**2).sum()),
               small_primes={str(int(p)): dict(fitted=fit1[i].tolist(), se=se1[i].tolist(), predicted=gm1[i].tolist(), predicted_with_dual=gf1[i].tolist()) for i, p in enumerate(pr_reg[:6])})
    print('%-5s regression p <= %d: slope chi %.4f +- %.4f (diagonal) / %.4f +- %.4f (with the dual term), chi^2 %.4f / %.4f; chi^2/coef %.2f / %.2f (chi), %.2f / %.2f (chi^2), %d each; slope by mode with dual %s' % (
          f, PREGR, *reg['slope_chi'], *reg['slope_chi_with_dual'], reg['slope_chi2'][0], reg['slope_chi2_with_dual'][0], reg['chi2_chi']/zz1.size, reg['chi2_chi_with_dual']/zz1.size,
          reg['chi2_chi2']/zz2.size, reg['chi2_chi2_with_dual']/zz2.size, zz1.size, ' '.join('%.3f' % a for a, b in slope_fk)), round(time.time() - t0), 's', flush=True)
    del F_, res
    # ---- (2) prediction y^ for p <= 1000 without a fit
    Yd = np.zeros((n, 7)); Yh = np.zeros((n, 7))
    for i, p in enumerate(pr_hat):
        x = kron(Ds, int(p)); x2 = x*x; m1, m2 = x.mean(), x2.mean(); Yd += gam(TB1, i)*(x - m1)[:, None] + gam(TB2, i)*(x2 - m2)[:, None]; Yh += gam(TF1, i)*(x - m1)[:, None] + gam(TF2, i)*(x2 - m2)[:, None]
    del TB1, TB2, TF1, TF2
    Yc = Y - ybar; expl_d = 1 - ((Yc - Yd)**2).sum(0)/(Yc**2).sum(0); corr_d = [float(np.corrcoef(Yc[:, k], Yd[:, k])[0, 1]) for k in range(7)]
    expl = 1 - ((Yc - Yh)**2).sum(0)/(Yc**2).sum(0); corr = [float(np.corrcoef(Yc[:, k], Yh[:, k])[0, 1]) for k in range(7)]
    print('%-5s explained share without a fit (p <= %d): diagonal %s, with the dual term %s; correlation %s / %s' % (f, PHAT, np.round(expl_d, 3).tolist(), np.round(expl, 3).tolist(), np.round(corr_d, 3).tolist(), np.round(corr, 3).tolist()), round(time.time() - t0), 's', flush=True)
    # ---- (3) stratifications: layer 0 from the characters alone against the exact layer 0 from the zeros
    def strat(V, M):
        tr = Sg*M + (Ds % M); uq, inv = np.unique(tr, return_inverse=True); Nc = np.bincount(inv); t_ = N*Nc/n; fl = np.floor(t_).astype(int)
        fl[np.argsort(-(t_ - fl), kind='stable')[:N - fl.sum()]] += 1
        s1 = np.zeros((len(uq), V.shape[1])); np.add.at(s1, inv, V); s2 = np.zeros_like(s1); np.add.at(s2, inv, V*V); mc = s1/Nc[:, None]
        S2c = np.where(Nc[:, None] > 1, (s2 - Nc[:, None]*mc*mc)/np.maximum(Nc - 1, 1)[:, None], 0.0)
        return ((fl[:, None]*mc).sum(0)/N)**2 + (fl[:, None]*(1 - fl[:, None]/Nc[:, None])*S2c).sum(0)/N**2
    E0r_h = Yh.mean(0)**2 + Yh.var(0, ddof=1)/N*(1 - N/n); E0r_d = Yd.mean(0)**2 + Yd.var(0, ddof=1)/N*(1 - N/n); ST = {}
    for M in ALLM:
        ex = strat(Y, M); eh = strat(Yh, M); ed = strat(Yd, M); pred = E0r - (E0r_h - eh)
        ST['mod%d' % M] = dict(exact=float(ex.sum()/80), predicted=float(pred.sum()/80), reduction_exact=float((E0r - ex).sum()/80), reduction_predicted=float((E0r_h - eh).sum()/80),
                               reduction_predicted_diagonal=float((E0r_d - ed).sum()/80), exact_k=(ex/80).tolist(), predicted_k=(pred/80).tolist())
    print('%-5s stratifications, reduction of layer 0 exact / predicted with the dual term / diagonal only: %s' % (f, ' '.join('%d %.5f/%.5f/%.5f' % (M, ST['mod%d' % M]['reduction_exact'], ST['mod%d' % M]['reduction_predicted'], ST['mod%d' % M]['reduction_predicted_diagonal']) for M in (8, 120, 360, 840))), round(time.time() - t0), 's', flush=True)
    # ---- (4) one stretch over every starting position
    def win(V):
        C = np.vstack([np.zeros(V.shape[1]), np.cumsum(V[order], 0)]); w = (C[N:] - C[:-N])/N; return (w**2).mean(0)
    ex_st = win(Y); eh_st = win(Yh); red_exact_hat = (E0r_h - eh_st); red_exact_hat_d = (E0r_d - win(Yd))
    # diagonal sum: every prime separately (family mean of gamma^2), checked against the exact window variance of y^ for p <= 1000
    jm_ = float(np.median(L)); kap_w = {0: float(np.mean(Sg == 0)), 1: float(np.mean(Sg == 1))}; Lq = np.quantile(L, [0.1, 0.3, 0.5, 0.7, 0.9])
    prd = PRI[PRI <= PDIAG]; GG = {key: [np.zeros((len(prd), 7)) for _ in range(3)] for key in ('d', 'f')}
    for lq in Lq:
        for kap in (0, 1):
            a1, a2, f1, f2 = weights(float(lq), kap, prd, seP); w_ = kap_w[kap]/len(Lq)
            for key, (b1_, b2_) in (('d', (a1, a2)), ('f', (f1, f2))): GG[key][0] += w_*b1_**2; GG[key][1] += w_*b2_**2; GG[key][2] += w_*b1_*b2_
    red = {key: [np.zeros(7), np.zeros(7)] for key in ('d', 'f')}; prof = []
    Do = Ds[order]
    for i, p in enumerate(prd):
        x = kron(Do, int(p)); x2 = x*x; v = []
        for z in (x, x2):
            mz = z.mean(); cz = np.concatenate([[0.0], np.cumsum(z)]); wz = (cz[N:] - cz[:-N])/N - mz; v.append((float(z.var()/N*(1 - N/n)), float((wz**2).mean()), wz))
        cov12 = float((v[0][2]*v[1][2]).mean()); covr = float(((x - x.mean())*(x2 - x2.mean())).mean()/N*(1 - N/n))
        for key in ('d', 'f'):
            G2, G22, G12 = GG[key]; r_ = G2[i]*(v[0][0] - v[0][1]) + G22[i]*(v[1][0] - v[1][1]) + 2*G12[i]*(covr - cov12); red[key][0] += r_
            if p <= PHAT: red[key][1] += r_
        if p in (2, 3, 5, 7, 11, 101, 1009, 10007, 100003): prof.append([int(p), v[0][1]/v[0][0] if v[0][0] > 0 else None])
    STR = dict(starts=int(n - N + 1), exact_from_zeros=float(ex_st.sum()/80), random_layer0=float(E0r.sum()/80), reduction_exact=float((E0r - ex_st).sum()/80),
               reduction_hat_p1000_exact=float(red_exact_hat.sum()/80), reduction_hat_p1000_exact_diagonal=float(red_exact_hat_d.sum()/80),
               reduction_sum_p1000=float(red['f'][1].sum()/80), reduction_sum=float(red['f'][0].sum()/80), reduction_sum_p1000_diagonal=float(red['d'][1].sum()/80), reduction_sum_diagonal=float(red['d'][0].sum()/80),
               predicted=float((E0r - red['f'][0]).sum()/80), reduction_exact_k=((E0r - ex_st)/80).tolist(), reduction_sum_k=(red['f'][0]/80).tolist(), reduction_sum_diagonal_k=(red['d'][0]/80).tolist(), r_p_sample=prof)
    print('%-5s stretch: layer 0 from the zeros %.5f, random %.5f; reduction exact %.5f, predicted (p <= %d, prime by prime) with the dual term %.5f, diagonal only %.5f  [p <= 1000: exact y^ %.5f / %.5f, prime by prime %.5f / %.5f]' % (
          f, STR['exact_from_zeros'], STR['random_layer0'], STR['reduction_exact'], PDIAG, STR['reduction_sum'], STR['reduction_sum_diagonal'], STR['reduction_hat_p1000_exact'], STR['reduction_hat_p1000_exact_diagonal'],
          STR['reduction_sum_p1000'], STR['reduction_sum_p1000_diagonal']), round(time.time() - t0), 's', flush=True)
    OUT['families'][f] = dict(n=n, l=float(L.mean()), random_layer0=float(E0r.sum()/80), windows=WIN, windows_summary=WINS, regression=reg, explained_without_fit=expl.tolist(), correlation=corr,
                              explained_without_fit_diagonal=expl_d.tolist(), correlation_diagonal=corr_d, stratifications=ST, stretch=STR)
    del Y, Yh, Yd, Yc
json.dump(OUT, open(os.path.join(OUT_DIR, 'reference_explicit_weights%s.json' % ('_' + '_'.join(ONLY) if ONLY else '')), 'w')); print('done', round(time.time() - t0), 's', flush=True)
