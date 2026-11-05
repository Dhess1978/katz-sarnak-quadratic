#!/usr/bin/env python3
"""
Study III, section 4.4: the mechanism without zeros (reference_characters.json).
(a) For every subset type of compute_moduli.py (random samples, stratifications by (sign, D mod M), consecutive bands) the variance of the mean of chi_D(p) over the subset,
    p up to 1 000, from 300 independent draws (own Kronecker symbol, Euler criterion; seed 34682) - W = weighted share of this variance left to the subset relative to a random
    sample, weight (ln p)^2/p from the explicit formula (W100: p up to 100, W1000: p up to 1 000).
(b) The profile of consecutive bands: r_p = variance of the mean of chi_D(p) over bands of 6 074 neighbouring D / variance of a random sample (finite population),
    for 582 primes up to 4·10^6 (all up to 2 000, then 4 primes after each of 70 logarithmic points; vectorised Jacobi symbol checked against the Euler criterion).
(d) The exact variance of stratified sampling without replacement (no draws) for all 20 stratifications (the sign and 19 moduli) and the bands, primes up to 10^4:
    W100 (p up to 100), W1000 and WE (weights (ln p)^2/p over all primes of the low modes, p up to e^(7L/8); above 10^4 the bands by the profile r_p).
(c) With reference_moduli.json: the line through two points (random samples at W = 1, consecutive bands) against the low-mode power of all 60 stratified subsets,
    chi^2 with the covariance of the two shared points; the slope from the stratifications alone; W_k = sum_k k r(e^(kL/8)) / 28 for the conductor dependence.
Only the discriminants are read; the zeros are not needed for (a) and (b).

usage:  python3 -I character_sums.py ZEROS_ROOT reference_moduli.json reference_characters.json      (about 15 minutes)
"""
import sys, os, json, math, time, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
t0 = time.time(); ROOT, MODP, OUTP = sys.argv[1], sys.argv[2], sys.argv[3]
RF = RZ.families(ROOT); DS = {}
for jm in ('B10', 'C100', 'D100'): DS[jm] = np.array([D for D, _ in RZ.zeros(jm, ROOT)], dtype=np.int64)
print('discriminants', {k: len(v) for k, v in DS.items()}, round(time.time() - t0), 's', flush=True)
NPAS = 6074; PMAX = 1000; NREP = 300
s = np.ones(PMAX + 1, bool); s[:2] = False
for p in range(2, int(PMAX**0.5) + 1):
    if s[p]: s[p*p::p] = False
PR = np.nonzero(s)[0]; W = np.log(PR)**2/PR
def kron(D, p):                       # Kronecker symbol (D/p) for a prime p, D an array of fundamental discriminants
    if p == 2:
        m = np.mod(D, 8); return np.where(D % 2 == 0, 0, np.where((m == 1) | (m == 7), 1, -1)).astype(np.int8)
    tab = np.array([0] + [1 if pow(r, (p - 1)//2, p) == 1 else -1 for r in range(1, p)], dtype=np.int8)   # Euler criterion
    return tab[np.mod(D, p)]
OUT = dict(study='III', version='34.68', npas=NPAS, pmax=PMAX, nrep=NREP, families={})
rng = np.random.default_rng(34682)
TYPES = ['random', 'mod1', 'mod8', 'mod16', 'mod3', 'mod7', 'mod41', 'mod120', 'mod360', 'mod520', 'mod840', 'mod131']   # mod131 last: the earlier draws stay unchanged; mod1 = sign only
for jm in ('B10', 'C100', 'D100'):
    Ds = DS[jm]; n_all = len(Ds); S = (Ds < 0).astype(int)
    X = np.stack([kron(Ds, int(p)) for p in PR]).astype(np.float32)            # (π(1000), n)
    mu = X.mean(1); var1 = X.var(1)
    def stat(ii):
        return (X[:, ii].mean(1) - mu)
    poradi = np.argsort(np.abs(Ds), kind='stable'); nP = n_all//NPAS
    res = {}
    dev = np.stack([stat(poradi[j*NPAS:(j + 1)*NPAS]) for j in range(nP)]); res['consecutive'] = (dev**2).mean(0)
    for typ in TYPES:
        devs = []
        if typ == 'random':
            for _ in range(NREP): devs.append(stat(rng.permutation(n_all)[:NPAS]))
        else:
            M = int(typ[3:]); tr = S*M + (Ds % M); uq, inv = np.unique(tr, return_inverse=True); Nc = np.bincount(inv); t = NPAS*Nc/n_all; fl = np.floor(t).astype(int)
            zb = NPAS - fl.sum(); fl[np.argsort(-(t - fl), kind='stable')[:zb]] += 1; cl = [np.nonzero(inv == c)[0] for c in range(len(uq))]
            for _ in range(NREP): devs.append(stat(np.concatenate([rng.choice(cl[c], fl[c], replace=False) for c in range(len(uq)) if fl[c] > 0])))
        res[typ] = (np.stack(devs)**2).mean(0)
    teor = var1/NPAS*(1 - NPAS/n_all)                                          # finite population, sampling without replacement
    R = {}
    for typ, v in res.items():
        R[typ] = dict(W100=float((W*v)[PR <= 100].sum()/(W*res['random'])[PR <= 100].sum()), W1000=float((W*v).sum()/(W*res['random']).sum()),
                      small_primes=[[int(p), float(v[i]/res['random'][i])] for i, p in enumerate(PR[:8])])
    OUT['families'][jm] = dict(n=n_all, bands=nP, random_over_theory=float(np.median(res['random']/teor)), types=R)
    print(jm, n_all, 'random / theory (median over p) %.4f' % np.median(res['random']/teor), round(time.time() - t0), 's', flush=True)
    for typ in ['consecutive'] + TYPES:
        r = R[typ]; print('  %-11s W(p<=100) %.4f  W(p<=1000) %.4f  ' % (typ, r['W100'], r['W1000']) + ' '.join('%d:%.3f' % (p, q) for p, q in r['small_primes']), flush=True)
    del X
def jacobi(a, n):                       # (a/n) for an array a >= 0 and an odd n > 0 (scalar)
    a = np.mod(a, n).astype(np.int64); nn = np.full(a.shape, n, np.int64); t = np.ones(a.shape, np.int64); act = a != 0
    while act.any():
        while True:
            ev = act & (a % 2 == 0)
            if not ev.any(): break
            a[ev] //= 2; r8 = nn[ev] % 8; t[ev] = np.where((r8 == 3) | (r8 == 5), -t[ev], t[ev])
        sw = act; a2 = a.copy(); a[sw], nn[sw] = nn[sw], a2[sw]
        q = sw & (a % 4 == 3) & (nn % 4 == 3); t[q] = -t[q]
        a[sw] = a[sw] % nn[sw]; act = a != 0
    return np.where(nn == 1, t, 0)
def kron2(D, p):
    if p == 2: m = np.mod(D, 8); return np.where(D % 2 == 0, 0, np.where((m == 1) | (m == 7), 1, -1))
    return jacobi(D, p)
def prvocisla(lo, hi):
    s = np.ones(hi - lo + 1, bool); s[:max(0, 2 - lo)] = False
    for p in range(2, int(hi**0.5) + 1):
        st = max(p*p, ((lo + p - 1)//p)*p); s[st - lo::p] = False
    return np.nonzero(s)[0] + lo
P_all = prvocisla(2, 2000)
VYB = list(P_all)
for x in np.exp(np.linspace(math.log(2000), math.log(4.0e6), 70)):      # logarithmic grid: 4 primes after each point
    VYB += list(prvocisla(int(x), int(x) + 400)[:4])
VYB = np.array(sorted(set(int(p) for p in VYB)))
PROF = dict(primes=VYB.tolist(), families={})
Dt = np.arange(-5000, 5001); chk = 0
for p in P_all[1:80]:
    e = np.array([0 if d % p == 0 else (1 if pow(int(d) % int(p), (int(p) - 1)//2, int(p)) == 1 else -1) for d in Dt]); chk = max(chk, int(np.abs(e - kron2(Dt, int(p))).max()))
PROF['jacobi_check'] = chk; print('Jacobi against Euler: largest difference', chk, flush=True)
for jm in ('B10', 'C100', 'D100'):
    Ds = DS[jm]; n_all = len(Ds); por = np.argsort(np.abs(Ds), kind='stable'); Dp = Ds[por]; nP = n_all//NPAS
    sirka = [int(abs(Dp[(j + 1)*NPAS - 1]) - abs(Dp[j*NPAS])) for j in range(nP)]
    L = float(np.mean(np.log(np.abs(Ds)/math.pi))); rp = []
    for p in VYB:
        x = kron2(Dp, int(p)).astype(np.float64); mu = x.mean(); v1 = x.var(); bm = x[:nP*NPAS].reshape(nP, NPAS).mean(1)
        teor = v1/NPAS*(1 - NPAS/n_all); rp.append(float(((bm - mu)**2).mean()/teor) if teor > 0 else float('nan'))
    rp = np.array(rp); lp = np.log(VYB.astype(float))
    # W_ef: integral over ln p with the density of primes (sum_p f(p) ~ int f(e^y) e^y/y dy): w_p e^y/y = y -> weight y dy; r interpolated in ln p
    def Wef(ymax, ymin=math.log(2)):
        y = np.linspace(ymin, ymax, 4000); rr = np.interp(y, lp, rp); return float(np.trapezoid(y*rr, y)/np.trapezoid(y, y))
    o = dict(n=n_all, bands=nP, width_median=float(np.median(sirka)), L=L, r=[round(float(q), 5) for q in rp], Wef_78=Wef(7*L/8), Wef_L=Wef(L), Wef_100=Wef(math.log(100)),
             pmax_78=float(math.exp(7*L/8)))
    PROF['families'][jm] = o
    print(jm, n_all, 'bands', nP, 'width |D| %.0f' % o['width_median'], 'L %.2f' % L, 'p_max(7L/8) %.3g' % o['pmax_78'], 'W_ef(7L/8) %.4f  W_ef(L) %.4f  W_ef(100) %.4f' % (o['Wef_78'], o['Wef_L'], o['Wef_100']), round(time.time() - t0), 's', flush=True)
    for lo, hi in ((2, 100), (100, 1000), (1000, 1e4), (1e4, 1e5), (1e5, 1e6), (1e6, 5e6)):
        m = (VYB >= lo) & (VYB < hi)
        if m.any(): print('   p %8.0f – %8.0f  r = %.3f (n %d)' % (lo, hi, np.nanmedian(rp[m]), m.sum()), flush=True)
# ---------------------------------------------------------------- (d) exact variance of stratified sampling without replacement, all 20 stratifications (no draws)
# k_c of the N elements are drawn from class c of size n_c:  V_p = sum_c k_c (1 - k_c/n_c) S_c^2 / N^2 + (sum_c k_c m_c / N - mu)^2,  S_c^2 with divisor n_c - 1.
ALLM = [1, 3, 5, 7, 8, 9, 13, 16, 24, 40, 41, 120, 130, 131, 260, 360, 369, 520, 800, 840]; PX = prvocisla(2, 10000); WX = np.log(PX)**2/PX
EXACT = {}
for jm in ('B10', 'C100', 'D100'):
    Ds = DS[jm]; n_all = len(Ds); Sg = (Ds < 0).astype(np.int64); por = np.argsort(np.abs(Ds), kind='stable'); nP = n_all//NPAS; ST = {}
    for M in ALLM:
        uq, inv = np.unique(Sg*M + Ds % M, return_inverse=True); Nc = np.bincount(inv); t = NPAS*Nc/n_all; k = np.floor(t).astype(np.int64)
        k[np.argsort(-(t - k), kind='stable')[:NPAS - int(k.sum())]] += 1; ST[M] = (inv, Nc, k)
    th = np.empty(len(PX)); vb = np.empty(len(PX)); V = {M: np.empty(len(PX)) for M in ALLM}
    for i, p in enumerate(PX):
        x = kron(Ds, int(p)).astype(np.float64); mu = x.mean(); th[i] = x.var()/NPAS*(1 - NPAS/n_all); bm = x[por][:nP*NPAS].reshape(nP, NPAS).mean(1); vb[i] = ((bm - mu)**2).mean()
        for M, (inv, Nc, k) in ST.items():
            s1 = np.bincount(inv, weights=x); s2 = np.bincount(inv, weights=x*x); m = s1/Nc; S2 = np.where(Nc > 1, (s2 - Nc*m*m)/np.maximum(Nc - 1, 1), 0.0)
            V[M][i] = np.sum(k*(1 - k/Nc)*S2)/NPAS**2 + (np.sum(k*m)/NPAS - mu)**2
    L = PROF['families'][jm]['L']; PL = math.exp(7*L/8); P_ = np.array(PROF['primes'], float); rr = np.array(PROF['families'][jm]['r']); th0 = float(np.median(th[PX > 1000]))
    def beyond(band):                         # primes 10^4 ... e^(7L/8): sum_p (ln p)^2/p f(p) ~ int y f(e^y) dy; stratifications f = 1, bands f = r_p of the profile
        if PL <= 10000: return 0.0
        y = np.linspace(math.log(10000), math.log(PL), 4000); fy = np.interp(y, np.log(P_), rr) if band else np.ones_like(y); return float(np.trapezoid(y*fy, y))
    def shares(v, band=False):
        a = PX <= 100; c = PX <= 1000; e = PX <= min(PL, 10000)
        return dict(W100=float(np.sum((WX*v)[a])/np.sum((WX*th)[a])), W1000=float(np.sum((WX*v)[c])/np.sum((WX*th)[c])), WE=float((np.sum((WX*v)[e]) + th0*beyond(band))/(np.sum((WX*th)[e]) + th0*beyond(False))))
    fixed = lambda M, p: (M % 8 == 0) if p == 2 else (M % p == 0)
    E = {'consecutive': shares(vb, True)}
    for M in ALLM:
        r = V[M]/th; fx = [r[i] for i, p in enumerate(PX) if p <= 1000 and fixed(M, p)]; ot = [r[i] for i, p in enumerate(PX) if p <= 1000 and not fixed(M, p)]
        E['mod%d' % M] = dict(**shares(V[M]), fixed_max=float(max(fx) if fx else 0.0), other=[float(min(ot)), float(max(ot))])
    EXACT[jm] = dict(L=L, p_low_modes=PL, types=E)
    print(jm, 'exact: bands W100 %.3f WE %.3f; mod 840 W100 %.3f; p | M at most %.3f, other primes %.3f-%.3f' % (E['consecutive']['W100'], E['consecutive']['WE'], E['mod840']['W100'],
          max(E['mod%d' % M]['fixed_max'] for M in ALLM), min(E['mod%d' % M]['other'][0] for M in ALLM), max(E['mod%d' % M]['other'][1] for M in ALLM)), round(time.time() - t0), 's', flush=True)
# ---------------------------------------------------------------- (c) line through two points with the exact W and the joint error of the two points, slope from the stratifications alone
from scipy.stats import chi2 as CHI2
MOD = json.load(open(MODP)); SUM = dict(families={}); DRAWN = {1, 3, 7, 8, 16, 41, 120, 131, 360, 520, 840}
for jm in ('B10', 'C100', 'D100'):
    T = {}
    for typ, v in MOD['families'][jm]['types'].items():
        a = np.array([q for ps_ in v['passes'].values() for q in ps_]); T[typ] = dict(n=len(a), low=float(a.mean()), se=float(a.std(ddof=1)/math.sqrt(len(a))))
        if typ in EXACT[jm]['types']: T[typ].update({k: EXACT[jm]['types'][typ][k] for k in ('W100', 'W1000', 'WE')})
        if typ in OUT['families'][jm]['types']: T[typ].update(W100_draws=OUT['families'][jm]['types'][typ]['W100'], W1000_draws=OUT['families'][jm]['types'][typ]['W1000'])
    T['random'].update(W100=1.0, W1000=1.0, WE=1.0)
    r_, s_ = T['random'], T['consecutive']
    for key, t in T.items():
        t['z'] = 0.0 if key == 'random' else (t['low'] - r_['low'])/math.sqrt(t['se']**2 + r_['se']**2)
        if key in ('random', 'consecutive'): continue
        for W in ('W100', 'WE'):
            q = (1 - t[W])/(1 - s_[W]); t['pred_' + W] = r_['low']*(1 - q) + s_['low']*q
            t['zp_' + W] = (t['low'] - t['pred_' + W])/math.sqrt(t['se']**2 + r_['se']**2*(1 - q)**2 + s_['se']**2*q*q)
    P_ = np.array(PROF['primes'], float); rr = np.array(PROF['families'][jm]['r']); L = PROF['families'][jm]['L']; Wk = [float(np.interp(k*L/8, np.log(P_), rr)) for k in range(1, 8)]
    po = s_['low']/r_['low']; se = po*math.sqrt((s_['se']/s_['low'])**2 + (r_['se']/r_['low'])**2)
    # slope alpha from the 20 stratifications alone (line through the random point; covariance: own errors + the shared error of the random samples)
    Pm = [T['mod%d' % M] for M in ALLM]; x = -np.array([1 - t['W100'] for t in Pm]); y = np.array([t['low'] for t in Pm]) - r_['low']; C = np.diag([t['se']**2 for t in Pm]) + r_['se']**2
    Ci = np.linalg.inv(C); al = float(x @ Ci @ y/(x @ Ci @ x)); al_se = float(1/math.sqrt(x @ Ci @ x)); res = y - al*x
    al_line = (r_['low'] - s_['low'])/(1 - s_['W100']); al_line_se = math.sqrt(r_['se']**2 + s_['se']**2)/(1 - s_['W100']); share = al/al_line
    SUM['families'][jm] = dict(types=T, Wk=Wk, W_k_weighted=float(sum(k*w for k, w in zip(range(1, 8), Wk))/28), band_ratio=po, band_ratio_se=se,
                               slope=dict(alpha=al, alpha_se=al_se, chi2=float(res @ Ci @ res), n=len(Pm), alpha_line=al_line, alpha_line_se=al_line_se, share=share, share_se=abs(share)*math.sqrt((al_se/al)**2 + (al_line_se/al_line)**2)))
    print(jm, 'W_k %.3f  band / random %.3f +- %.3f; slope from stratifications %.4f +- %.4f, line %.4f -> share of the band deficit from p <= 100: %.2f +- %.2f' % (
          SUM['families'][jm]['W_k_weighted'], po, se, al, al_se, al_line, share, SUM['families'][jm]['slope']['share_se']), flush=True)
def line(W, sel):
    tot, nn, zz = 0.0, 0, []
    for jm, F in SUM['families'].items():
        T = F['types']; r_, s_ = T['random'], T['consecutive']; P = [T['mod%d' % M] for M in ALLM if sel(M)]
        q = np.array([(1 - t[W])/(1 - s_[W]) for t in P]); d = np.array([t['low'] - t['pred_' + W] for t in P])
        C = np.diag([t['se']**2 for t in P]) + r_['se']**2*np.outer(1 - q, 1 - q) + s_['se']**2*np.outer(q, q); tot += float(d @ np.linalg.solve(C, d)); nn += len(P); zz += [t['zp_' + W] for t in P]
    return dict(chi2=tot, n=nn, p=float(CHI2.sf(tot, nn)), positive=int(sum(1 for z in zz if z > 0)), mean_z=float(np.mean(zz)), max=float(np.max(np.abs(zz))))
SUM['line'] = dict(W100_60=line('W100', lambda M: True), W100_33=line('W100', lambda M: M in DRAWN), WE_60=line('WE', lambda M: True))
nu = sum(F['slope']['share']/F['slope']['share_se']**2 for F in SUM['families'].values()); de = sum(1/F['slope']['share_se']**2 for F in SUM['families'].values()); SUM['share'] = dict(share=nu/de, se=1/math.sqrt(de))
for k, v in SUM['line'].items(): print(k, 'line through two points: chi2 %.2f / %d (p %.3f), positive deviations %d, mean z %+.2f, max |z| %.2f' % (v['chi2'], v['n'], v['p'], v['positive'], v['mean_z'], v['max']), flush=True)
print('share of the band deficit carried by p <= 100, jointly %.2f +- %.2f' % (SUM['share']['share'], SUM['share']['se']))
json.dump(dict(sums=OUT, profile=PROF, exact=EXACT, summary=SUM), open(OUTP, 'w')); print('done', round(time.time() - t0), 's', flush=True)
