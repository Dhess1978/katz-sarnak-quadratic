#!/usr/bin/env python3
"""
Independent check of study III (numpy, scipy, mpmath; cypari2 only for lfunhardy - a different PARI function from lfunzeros used by the generator).
Shares no code with compute_families.py or read_zeros.py: own reader, own unfolding, own prediction, own cosine matrix, own Fredholm quadrature,
own random draws.

usage:  python3 -I independent_check.py ZEROS_ROOT reference_families.json reference_subsets.json reference_pieces.json [reference_moduli.json reference_characters.json [reference_pieces_repeat.json
        [reference_layers.json [reference_intermodulation.json reference_first_zero_crossing.json [reference_recomputation.json reference_two_level.json reference_missing_zeros_B10.json [reference_families_E100.json]]]]]]
        (about 4 minutes on two cores; checks 9 and 10 of section 4.4 only with the next two files, check 11 of section 4.6 only with reference_layers.json, checks 12 and 13 with the next two,
        checks 14-16 of section 4.7 with the next three, check 17 of section 4.8 with reference_families_E100.json)

Checks (each printed PASS/FAIL):
 1  composition: B10, C100, D100, E10 by an own sieve for squarefree numbers, R10 by an own splitmix64 = the discriminants in the files
 2  histograms of the unfolded zeros (own Stirling series) = reference, for every family and sign
 3  completeness: 8 L-functions per family - sign changes of the Hardy function (lfunhardy) on a fine grid = number of stored zeros, and a sign change at every zero
 4  prediction of the ratios conjecture by another route (pole terms subtracted analytically, cubic splines, Gauss-Legendre in each bin) = reference within 2e-3
 5  consecutive bands (own sorting by |D|): histograms identical
 6  chi^2 and low-frequency power of own random samples (own seed) - ratio consecutive / random
 7  first zero: own N_eff by bisection, Fredholm determinant Gauss-Legendre 60; even and odd characters; fit c/l of the even deviation
 8  pieces: own random positions in C100, m = 1, 10, 100 and random - low-frequency power increases with m; B10 at m = 300, 1 000, 2 000 and random with
    300 own subsets per point and the error of the mean - no step down beyond 2 errors, and the repetition of reference_pieces_repeat.json within 3 joint errors
 9  character sums without zeros: own chi_D(p) from the squares mod p, exact variance of a stratified sample without replacement (no draws), checked against own draws;
    (sign, D mod 840) fixes only p = 2, 3, 5, 7; the band profile r_p for p < 100 = reference; reference random draws = finite population
10  the line through two points (random samples, consecutive bands) with the exact W100 of all 20 stratifications and the low-mode power of the zeros from reference_moduli.json,
    chi^2 with the covariance of the two shared points (own linear algebra)
11  exact expectations (Section 4.6): own contributions y_ik of every L-function in B10 (own prediction for every discriminant, Pearson error from the expected count),
    layer 0 of the random sample, of triples and of one stretch without draws; own draws of the Pearson statistic = layer 0; the test statistic (observed count) above it,
    the difference carried by the two lowest bins; coupling of the modes 1-7; regression on own Legendre symbols p <= 30 - all against reference_layers.json
12  intermodulation (Section 4.6): own y_ik in B10, regression on own Legendre symbols p <= 29 and on chi_D(n) of the 16 squarefree composite n < e^4 built from them;
    the gain of the modes 6-7 is many times that of the modes 1-4, the same features permuted by an own generator give almost nothing - against reference_intermodulation.json
13  crossing of the first zero (Section 4.5): own first zeros, own N_eff and Fredholm GL 60 on a grid of 0.001 in C100 and B10; crossing of the even curve with zero,
    of the even and the odd curve and the return into the Kolmogorov-Smirnov band = reference_first_zero_crossing.json; a pure shift of the model gives no crossing;
    calibration test: with N_eff fixed on [0, 0.4] and [0, 0.6] instead of [0, 1/2] the crossing in C100 moves with the end of the interval
14  recomputation by layers (Section 4.7): own exact layer 0 of the random sample and of (sign, D mod 120) and (sign, D mod 840) in B10 from the own y_ik of check 11, own allocation
    by largest remainders and the exact variance of stratified sampling; the line through two points with the values by layers and the own W100 of check 9 (rejected); layer-0 check
15  first zero (Section 4.7): own eigenvalues of the kernel S_N(x - y) - S_N(x + y) (GL 60), E[N(N - 1)] at a = 1/2 in the model and in the own unfolded data of C100;
    own N_eff on [0, 0.3] and [0, 0.7] and the mean first zero against the model for both signs
16  completeness (Section 4.7): own Hardy function (lfunhardy) on a grid of 0.0005 around the close pair of zeros of the example in reference_missing_zeros_B10.json
17  the family E100 (Section 4.8): own sieve of the fundamental discriminants with 10^10 < |D| < 10^10 + 2*10^5 = the files of E10 and E100; own unfolding,
    own N_eff and Fredholm GL 60 for the mean first zero of both signs = reference_families_E100.json; odd characters within 2 errors, even below 0.094/l
"""
import sys, os, json, math, gzip, glob, time
import numpy as np, mpmath as mp, scipy.special as sp
from scipy.interpolate import CubicSpline
from scipy.integrate import quad
t0 = time.time(); ROOT, RF, RS, RP = sys.argv[1:5]; REF = json.load(open(RF)); SUB = json.load(open(RS)); PCS = json.load(open(RP)); OK = []
def check(name, cond, info): OK.append(bool(cond)); print('%s  %d %s: %s' % ('PASS' if cond else 'FAIL', len(OK), name, info), flush=True)
XF = 0.025; NF = 160; XB = 0.05; NB = 80; NP = 6074
FAMS = ['B10', 'C100', 'D100', 'E10', 'R10']
def find_set(fam):
    for d in sorted(os.listdir(ROOT)):
        m = os.path.join(ROOT, d, 'manifest.json')
        if os.path.isfile(m) and fam in json.load(open(m))['rodiny'] and os.path.isdir(os.path.join(ROOT, d, fam)): return os.path.join(ROOT, d)
    raise SystemExit('family %s not found' % fam)
def load(fam):
    d = find_set(fam); Ds, G = [], []
    for f in sorted(glob.glob(os.path.join(d, fam, 'blok_*.jsonl.gz'))):
        for line in gzip.open(f, 'rt'): o = json.loads(line); Ds.append(o['D']); G.append(o['g'])
    return Ds, G, json.load(open(os.path.join(d, 'manifest.json')))['rodiny'][fam]['spec']
DATA = {f: load(f) for f in FAMS}
# ---- 1) composition
def sqf_range(lo, hi):
    f = np.ones(hi - lo, bool); P = int(math.isqrt(hi)) + 1; s = np.ones(P + 1, bool); s[:2] = False
    for p in range(2, int(math.isqrt(P)) + 1):
        if s[p]: s[p*p::p] = False
    for p in np.nonzero(s)[0]:
        q = int(p)*int(p); st = -(-(lo + 1)//q)*q
        if st <= hi: f[st - lo - 1::q] = False
    return f
def fundamental_range(lo, hi):
    S = sqf_range(lo, hi); S4 = sqf_range(lo//4, hi//4 + 1); out = []
    for a in range(lo + 1, hi + 1):
        for D in (a, -a):
            if D % 4 == 1 and S[a - lo - 1]: out.append(D)
            elif D % 4 == 0 and (D//4) % 4 in (2, 3) and S4[abs(D//4) - lo//4 - 1]: out.append(D)
    return out
def splitmix(seed):
    s = seed % 2**64
    while True:
        s = (s + 0x9E3779B97F4A7C15) % 2**64; z = s; z = ((z ^ (z >> 30))*0xBF58476D1CE4E5B9) % 2**64; z = ((z ^ (z >> 27))*0x94D049BB133111EB) % 2**64; yield z ^ (z >> 31)
def sqf(m):
    m = abs(m)
    for p in range(2, int(math.isqrt(m)) + 1):
        if m % (p*p) == 0: return False
    return True
def fundamental(D): return (D % 4 == 1 and sqf(D)) or (D % 4 == 0 and (D//4) % 4 in (2, 3) and sqf(D//4))
def random_family(F):
    g = splitmix(F['seed']); out = set(); span = F['hi'] - F['lo']
    while len(out) < F['n']:
        r = next(g); a = F['lo'] + r % span; D = a if (r >> 63) & 1 else -a
        if fundamental(D): out.add(D)
    return out
res = {}
for f, (Ds, G, F) in DATA.items(): exp = set(fundamental_range(F['lo'], F['hi'])) if F['typ'] == 'rozsah' else random_family(F); res[f] = set(Ds) == exp and len(Ds) == len(set(Ds))
check('composition of the families', all(res.values()), res)
# ---- 2) histograms by own unfolding
def lg_im(a, t):
    z = complex(a, t); s = 0j
    while z.real < 15: s -= np.log(z); z += 1
    zi = 1/z; z2 = zi*zi
    return ((z - 0.5)*np.log(z) - z + 0.5*math.log(2*math.pi) + zi*(1/12 - z2*(1/360 - z2*(1/1260 - z2*(1/1680 - z2/1188)))) + s).imag
def th(D, t): k = 0 if D > 0 else 1; return 0.5*t*math.log(abs(D)/math.pi) + lg_im((0.5 + k)/2, t/2)
HH = {}; X1 = {}; LL = {}; bad = 0
for f, (Ds, G, F) in DATA.items():
    hh = np.zeros((len(Ds), NF), np.int16); x1 = np.full(len(Ds), np.nan)
    for i, (D, g) in enumerate(zip(Ds, G)):
        for j, v in enumerate(g):
            x = th(D, v)/math.pi
            if x < 4:
                hh[i, int(x/XF)] += 1
                if j == 0: x1[i] = x
    HH[f] = hh; X1[f] = x1; LL[f] = np.log(np.abs(np.array(Ds, float))/math.pi); S = np.array([0 if D > 0 else 1 for D in Ds])
    for sign, m in (('all', np.ones(len(Ds), bool)), ('plus', S == 0), ('minus', S == 1)):
        if hh[m].sum(0).astype(int).tolist() != REF['families'][f][sign]['hist_0025']: bad += 1
check('histograms by own unfolding = reference (15 family x sign)', bad == 0, 'mismatches %d' % bad)
# ---- 3) completeness by the Hardy function
import cypari2
pari = cypari2.Pari(); pari.allocatemem(64*2**20, 2*2**30, silent=True); comp = []
for f, (Ds, G, F) in DATA.items():
    for i in [int(i) for i in np.linspace(0, len(Ds) - 1, 8)]:
        D = Ds[i]; g = G[i]; T = g[-1] + 0.02; L = pari.lfuninit(pari.lfuncreate(D), pari('[1/2, 0, %.6f]' % (T + 0.1)))
        dg = min([b - a for a, b in zip(g, g[1:])] + [0.2]); st = dg/6; ts = np.arange(st/2, T, st)
        vals = [float(pari.lfunhardy(L, float(t))) for t in ts]; zm = sum(1 for a, b in zip(vals, vals[1:]) if a*b < 0)
        nz = sum(1 for v in g if v < ts[-1]); okz = all(float(pari.lfunhardy(L, v - 1e-7))*float(pari.lfunhardy(L, v + 1e-7)) < 0 for v in g)
        comp.append(nz == zm and okz)
check('completeness of the zeros (Hardy function, 40 L)', all(comp), '%d / %d' % (sum(comp), len(comp)))
# ---- 4) prediction by another route: G1 = zeta'/zeta(1+2it) + A' + 1/(2it), G2 = Gamma ratio * zeta(1-2it) * A_D + 1/(2it)
Mp = 100000; s = np.ones(Mp + 1, bool); s[:2] = False
for p in range(2, int(Mp**0.5) + 1):
    if s[p]: s[p*p::p] = False
P = np.nonzero(s)[0].astype(float); lp = np.log(P)
DT = 0.004; TG = np.arange(0, 8.2 + 1e-9, DT); TG[0] = 1e-7
G1 = np.zeros(len(TG), complex); ZR = np.zeros(len(TG), complex); AD = np.zeros(len(TG), complex); mp.mp.dps = 20
for i, t in enumerate(TG):
    z = np.exp(2j*t*lp); AD[i] = np.prod(1 - (z - 1)/(P*P - 1)); a2 = np.sum(lp/((P + 1)*(P*z - 1)))
    s1 = mp.mpc(1, 2*t); G1[i] = complex(mp.zeta(s1, derivative=1)/mp.zeta(s1)) + a2 + 1/(2j*t); ZR[i] = complex(mp.zeta(mp.mpc(1, -2*t)))
GR = {k: np.exp(sp.loggamma(0.25 + 0.5*k - 0.5j*TG) - sp.loggamma(0.25 + 0.5*k + 0.5j*TG)) for k in (0, 1)}
G2 = {k: GR[k]*ZR*AD + 1/(2j*TG) for k in (0, 1)}; S1 = CubicSpline(TG, G1.real); S2 = {k: (CubicSpline(TG, G2[k].real), CubicSpline(TG, G2[k].imag)) for k in (0, 1)}
def cl(l, k, t): e = np.exp(-1j*t*l); return (S1(t) - (e.real*S2[k][0](t) - e.imag*S2[k][1](t)) - np.sin(t*l)/(2*t))/math.pi
def thl(l, k, t): return 0.5*np.asarray(t)*l + np.imag(sp.loggamma(0.25 + 0.5*k + 0.5j*np.asarray(t)))
XQ, WQ = np.polynomial.legendre.leggauss(10); TF = np.linspace(1e-9, 8.1, 40001)
def predl(l, k, xb=XB, nb=NB):
    te = np.interp(np.arange(nb + 1)*xb, thl(l, k, TF)/math.pi, TF)
    for _ in range(3): d = (0.5*l + 0.5*np.real(sp.digamma(0.25 + 0.5*k + 0.5j*te)))/math.pi; te = te - (thl(l, k, te)/math.pi - np.arange(nb + 1)*xb)/d
    te[0] = 0.0; a, b = te[:-1], te[1:]; m, h = 0.5*(a + b), 0.5*(b - a); T = np.maximum(m[:, None] + h[:, None]*XQ[None, :], 1e-7)
    return 1 + (cl(l, k, T) @ WQ)*h/xb
LS = 0.01; NODES = {}
def node(k, j):
    if (k, j) not in NODES: NODES[(k, j)] = predl(j*LS, k)
    return NODES[(k, j)]
def pred_mean(L, S):
    tot = np.zeros(NB)
    for k in (0, 1):
        li = L[S == k]
        if not len(li): continue
        j = np.floor(li/LS).astype(int); u = li/LS - j
        for jj in np.unique(j):
            m = j == jj; c = m.sum(); su = u[m].sum(); tot += (c - su)*node(k, int(jj)) + su*node(k, int(jj) + 1)
    return tot/len(L)
dmax = 0
for f in FAMS:
    Ds = np.array(DATA[f][0]); S = (Ds < 0).astype(int)
    for sign, k in (('plus', 0), ('minus', 1)):
        m = S == k; mine = pred_mean(LL[f][m], S[m]); ref = np.array(REF['families'][f][sign]['prediction_0025']).reshape(NB, 2).mean(1); dmax = max(dmax, float(np.max(np.abs(mine - ref))))
check('prediction by another route = reference (bins 0.05)', dmax < 2e-3, 'max %.1e' % dmax)
# ---- 5) subsets: consecutive bands identical; own random samples
CM = np.array([[math.sqrt((1 if k == 0 else 2)/NB)*math.cos(math.pi*k*(b + 0.5)/NB) for b in range(NB)] for k in range(NB)])
def subset(f, ii):
    hh = HH[f][ii].reshape(len(ii), NB, 2).sum(2).astype(float); H = hh.sum(0); n = len(ii); Ds = np.array(DATA[f][0])[ii]; L = LL[f][ii]; S = (Ds < 0).astype(int)
    pm = pred_mean(L, S); r = (H/(n*XB) - pm)/(np.sqrt(np.maximum(H, 1))/(n*XB)); c = CM @ r
    return H.astype(int).tolist(), float(np.mean(r*r)), float(np.sum(c[1:8]**2)/NB)
rng = np.random.default_rng(20261103); info = {}; same = True; ratios = {}
for f in ('B10', 'C100', 'D100'):
    Ds = np.array(DATA[f][0]); order = np.argsort(np.abs(Ds), kind='stable'); nP = len(Ds)//NP
    refS = [q for q in SUB['families'][f]['subsets'] if q['type'] == 'consecutive']; cS = []; lS = []
    for j in range(nP):
        h, c, lo = subset(f, np.sort(order[j*NP:(j + 1)*NP])); same &= (h == refS[j]['hist_005']); cS.append(c); lS.append(lo)
    cN = []; lN = []
    for _ in range({'B10': 81, 'C100': 100, 'D100': 30}[f]):
        h, c, lo = subset(f, np.sort(rng.permutation(len(Ds))[:NP])); cN.append(c); lN.append(lo)
    refN = [q['low_k1_7'] for q in SUB['families'][f]['subsets'] if q['type'] == 'random']; refSl = [q['low_k1_7'] for q in refS]
    ratios[f] = (np.mean(lS)/np.mean(lN), np.mean(refSl)/np.mean(refN)); se = np.std(lN, ddof=1)/math.sqrt(len(lN))
    info[f] = 'chi2 consecutive %.3f / own random %.3f; low k 1-7 consecutive %.4f / own random %.4f +- %.4f (reference random %.4f); ratio %.3f (reference %.3f)' % (np.mean(cS), np.mean(cN), np.mean(lS), np.mean(lN), se, np.mean(refN), ratios[f][0], ratios[f][1])
check('consecutive bands: histograms identical to the reference', same, '')
check('own random samples: low frequencies of consecutive bands below random in B10, C100, D100 (ratio within 25 % of the reference)', all(r[0] < 0.6 and abs(r[0]/r[1] - 1) < 0.25 for r in ratios.values()), info)
# ---- 6) first zero: own N_eff (USp(2N) density in [0, 1/2] by Gauss-Legendre), Fredholm GL 60
def W_N(x, N): y = np.pi*x/N; return 1 + 1/(2*N) - np.sin((2*N + 1)*y)/(2*N*np.sin(y))
XD, WD = np.polynomial.legendre.leggauss(40)
def d0_2n(N, a=0.5): x = 0.5*a*(XD + 1); return float(np.sum(WD*W_N(np.maximum(x, 1e-12), N))*0.5)
def n_eff(d0p, a=0.5):
    lo, hi = 0.55, 80.0
    for _ in range(70):
        mid = 0.5*(lo + hi)
        if d0_2n(mid, a) > d0p: lo = mid
        else: hi = mid
    return 0.5*(lo + hi)
X60, W60 = np.polynomial.legendre.leggauss(60)
def E0(s_, N):
    if s_ <= 0: return 1.0
    if s_ >= 0.98*N: return 0.0
    x = 0.5*s_*(X60 + 1); w = 0.5*s_*W60
    Sn = lambda z: np.where(np.abs(z) < 1e-12, (2*N + 1)/(2*N), np.sin((2*N + 1)*np.pi*z/(2*N))/(2*N*np.sin(np.pi*z/(2*N)) + 1e-300))
    K = Sn(x[:, None] - x[None, :]) - Sn(x[:, None] + x[None, :]); return float(np.linalg.det(np.eye(60) - np.sqrt(w)[:, None]*K*np.sqrt(w)[None, :]))
dev = {}; dm = 0; NEF = {}
for f in FAMS:
    Ds = np.array(DATA[f][0]); S = (Ds < 0).astype(int)
    for sign, k in (('plus', 0), ('minus', 1)):
        m = S == k; d0p = float(np.mean(pred_mean(LL[f][m], S[m])[:10])); Ne = n_eff(d0p); NEF[f + '|' + sign] = Ne; mef = quad(lambda s_: E0(s_, Ne), 0, min(3.0, 0.98*Ne), limit=200, epsabs=1e-7)[0]
        x1 = X1[f][m]; x1 = x1[np.isfinite(x1)]; dev[f + '|' + sign] = (float(LL[f][m].mean()), float(x1.mean() - mef), float(x1.std()/math.sqrt(len(x1))))
        dm = max(dm, abs(mef - REF['families'][f][sign]['x1_usp2n_eff']))
B = [dev[f + '|plus'] for f in FAMS]; Xl = np.array([1/b[0] for b in B]); dd = np.array([b[1] for b in B]); ss = np.array([b[2] for b in B]); c = float(np.sum(Xl*dd/ss**2)/np.sum(Xl*Xl/ss**2)); chi2 = float(np.sum(((dd - c*Xl)/ss)**2))
odd = [abs(dev[f + '|minus'][1]/dev[f + '|minus'][2]) for f in FAMS]
check('first zero: own N_eff and Fredholm GL 60 = reference; odd characters within 2 errors; even C100 > 5 errors above; fit c/l', dm < 3e-3 and max(odd) < 2 and dev['C100|plus'][1]/dev['C100|plus'][2] > 5 and chi2/4 < 2,
      'model means within %.1e; odd max %.1f errors; even C100 %.1f errors; c = %.4f (chi2 %.2f / 4)' % (dm, max(odd), dev['C100|plus'][1]/dev['C100|plus'][2], c, chi2))
# ---- 7) pieces with own random positions in C100
Ds = np.array(DATA['C100'][0]); order = np.argsort(np.abs(Ds), kind='stable'); n = len(Ds); low = {}
for m_ in (1, 10, 100, NP):
    v = []
    for _ in range(50):
        if m_ == NP: ii = rng.permutation(n)[:NP]
        else:
            dl = NP//m_; st = np.sort(rng.choice(n//(dl + 1), m_, replace=False))*(dl + 1); ii = np.concatenate([order[s_:s_ + dl + (1 if j < NP - dl*m_ else 0)] for j, s_ in enumerate(st)])
        v.append(subset('C100', np.sort(ii))[2])
    low[m_] = float(np.mean(v))
# B10 near the random end with the error of every point (own random positions, 300 subsets per m): the curve has no step down beyond 2 errors
Db = np.array(DATA['B10'][0]); ob = np.argsort(np.abs(Db), kind='stable'); nb_ = len(Db); lowB = {}; rngB = np.random.default_rng(20261104)
for m_ in (300, 1000, 2000, NP):
    v = []
    for _ in range(300):
        if m_ == NP: ii = rngB.permutation(nb_)[:NP]
        else:
            dl = NP//m_; st = np.sort(rngB.choice(nb_//(dl + 1), m_, replace=False))*(dl + 1); ii = np.concatenate([ob[s_:s_ + dl + (1 if j < NP - dl*m_ else 0)] for j, s_ in enumerate(st)])
        v.append(subset('B10', np.sort(ii))[2])
    lowB[m_] = (float(np.mean(v)), float(np.std(v, ddof=1)/math.sqrt(len(v))))
kB = list(lowB); stepB = min((lowB[kB[i + 1]][0] - lowB[kB[i]][0])/math.hypot(lowB[kB[i]][1], lowB[kB[i + 1]][1]) for i in range(len(kB) - 1))
repB = ''
if len(sys.argv) > 7:
    RP2 = json.load(open(sys.argv[7]))['repetition']['families']['B10']['m']; zr = max(abs(lowB[m_][0] - RP2[str(m_)]['power_k1_7'])/math.hypot(lowB[m_][1], RP2[str(m_)]['power_k1_7_se']) for m_ in kB)
    repB = '; reference repetition (400 subsets) within %.1f joint errors' % zr
else: zr = 0.0
check('pieces: low-frequency power grows with the number of pieces (C100); B10 with the error of every point has no step down (300 own subsets per m)', low[1] < low[10] < low[100] and low[1]/low[NP] < 0.5 and stepB > -2 and zr < 3,
      ' / '.join('m=%d %.4f' % kv for kv in low.items()) + '; B10 ' + ' / '.join('m=%d %.4f +- %.4f' % (m_, *lowB[m_]) for m_ in kB) + ', worst step %+.1f errors' % stepB + repB)
# ---- 8) and 9) section 4.4: own character values, exact variance of stratified sampling without replacement, the line through two points
if len(sys.argv) > 6:
    RMOD = json.load(open(sys.argv[5])); RCH = json.load(open(sys.argv[6]))
    P100 = [p for p in range(2, 100) if all(p % q for q in range(2, int(math.isqrt(p)) + 1))]; wt = np.array([math.log(p)**2/p for p in P100])
    SQ = {p: np.array([0] + [1 if r in set((x*x) % p for x in range(1, p)) else -1 for r in range(1, p)], np.int8) for p in P100 if p > 2}
    def chars(Ds):
        out = np.empty((len(P100), len(Ds)))
        for i, p in enumerate(P100):
            if p == 2: r8 = Ds % 8; out[i] = np.where(Ds % 2 == 0, 0, np.where((r8 == 1) | (r8 == 7), 1, -1))
            else: out[i] = SQ[p][Ds % p]
        return out
    def strat_alloc(cls):
        uq, inv = np.unique(cls, return_inverse=True); cnt = np.bincount(inv); q = NP*cnt/len(cls); k = np.floor(q).astype(np.int64)
        k[np.argsort(-(q - k), kind='stable')[:NP - int(k.sum())]] += 1; return inv, cnt, k
    def exact_var(X, mu, inv, cnt, k):
        """E[(mean of the sample - mean of the family)^2] for k[c] drawn without replacement from class c, every prime at once"""
        out = np.empty(len(X))
        for i in range(len(X)):
            s1 = np.bincount(inv, weights=X[i], minlength=len(cnt)); s2 = np.bincount(inv, weights=X[i]**2, minlength=len(cnt)); m = s1/cnt
            S2 = np.where(cnt > 1, (s2 - cnt*m*m)/np.maximum(cnt - 1, 1), 0.0); out[i] = np.sum(k*(1 - k/cnt)*S2)/NP**2 + (np.sum(k*m)/NP - mu[i])**2
        return out
    STRAT = [1, 3, 5, 7, 8, 9, 13, 16, 24, 40, 41, 120, 130, 131, 260, 360, 369, 520, 800, 840]; own = {}; prof_d = 0; info8 = {}; ok8 = list(RCH['profile']['primes'][:len(P100)]) == P100; rng8 = np.random.default_rng(4411)
    for f in ('B10', 'C100', 'D100'):
        Ds = np.array(DATA[f][0], np.int64); n = len(Ds); X = chars(Ds); mu = X.mean(1); theo = X.var(1)/NP*(1 - NP/n)
        order = np.argsort(np.abs(Ds), kind='stable'); nP = n//NP; bm = np.stack([X[:, order[j*NP:(j + 1)*NP]].mean(1) for j in range(nP)]); vb = ((bm - mu)**2).mean(0)
        prof_d = max(prof_d, float(np.max(np.abs(vb/theo - np.array(RCH['profile']['families'][f]['r'][:len(P100)])))))
        own[f] = {'consecutive': float(np.sum(wt*vb)/np.sum(wt*theo))}
        for M in STRAT:
            inv, cnt, k = strat_alloc((Ds < 0).astype(np.int64)*M + Ds % M); v = exact_var(X, mu, inv, cnt, k); own[f]['mod%d' % M] = float(np.sum(wt*v)/np.sum(wt*theo))
            if M == 840:
                fixed = (v/theo)[:4]; other = float(np.median((v/theo)[4:])); ok8 &= bool(fixed.max() < 0.1 and 0.7 < other < 1.4)
                info8[f] = 'mod 840: p = 2, 3, 5, 7 at most %.3f, other primes median %.2f; reference random / finite population %.4f' % (fixed.max(), other, RCH['sums']['families'][f]['random_over_theory'])
                if f == 'B10':                                            # the exact formula against own draws (200 stratified samples)
                    start = np.concatenate([[0], np.cumsum(cnt)[:-1]]); dv = []
                    for _ in range(200):
                        o = np.argsort(inv + rng8.random(n)); rank = np.empty(n, np.int64); rank[o] = np.arange(n) - start[inv[o]]; ii = np.nonzero(rank < k[inv])[0]; dv.append(X[:, ii].mean(1) - mu)
                    vd = (np.stack(dv)**2).mean(0); wdraw = float(np.sum(wt*vd)/np.sum(wt*theo)); ok8 &= abs(wdraw - own[f]['mod840']) < 0.08; info8[f] += '; W100 exact %.3f, own draws %.3f' % (own[f]['mod840'], wdraw)
        ok8 &= abs(RCH['sums']['families'][f]['random_over_theory'] - 1) < 0.01
    check('character sums without zeros (own squares mod p, exact stratified variance): mod 840 fixes only p = 2, 3, 5, 7; band profile r_p (p < 100) = reference', ok8 and prof_d < 1e-4, '%s; profile within %.1e' % (info8, prof_d))
    chi2 = 0.0; wd = 0.0; npts = 0; pos = 0
    for f in ('B10', 'C100', 'D100'):
        T = RMOD['families'][f]['types']; val = lambda t: np.array([x for ps in T[t]['passes'].values() for x in ps]); ms = lambda a: (a.mean(), a.std(ddof=1)/math.sqrt(len(a)))
        (R0, Rse), (S0, Sse) = ms(val('random')), ms(val('consecutive')); Ws = own[f]['consecutive']; wd = max(wd, abs(Ws - RCH['summary']['families'][f]['types']['consecutive']['W100']))
        q = np.array([(1 - own[f]['mod%d' % M])/(1 - Ws) for M in STRAT]); m = np.array([ms(val('mod%d' % M))[0] for M in STRAT]); se = np.array([ms(val('mod%d' % M))[1] for M in STRAT])
        d = m - (R0*(1 - q) + S0*q); Cm = np.diag(se**2) + Rse**2*np.outer(1 - q, 1 - q) + Sse**2*np.outer(q, q); chi2 += float(d @ np.linalg.solve(Cm, d)); npts += len(STRAT); pos += int(np.sum(d > 0))
        wd = max(wd, max(abs(own[f]['mod%d' % M] - RCH['summary']['families'][f]['types']['mod%d' % M]['W100']) for M in STRAT))
    ref = RCH['summary']['line']['W100_60']
    check('line through two points: exact W100 of all 60 stratifications (characters) against the low-mode power of their zeros, chi2 with the covariance of the two shared points (p > 0.01 for 60 degrees of freedom)',
          npts == 60 and chi2 < 88.38 and abs(chi2 - ref['chi2']) < 1e-3 and wd < 1e-6, 'chi2 %.1f / %d (reference %.1f), %d of 60 deviations positive; exact W100 = reference within %.1e' % (chi2, npts, ref['chi2'], pos, wd))
# ---- 11) section 4.6: exact expectations without draws, own route
if len(sys.argv) > 8:
    RL = json.load(open(sys.argv[8]))['families']['B10']; Ds = np.array(DATA['B10'][0], np.int64); n = len(Ds); L = LL['B10']; S = (Ds < 0).astype(int); order = np.argsort(np.abs(Ds), kind='stable')
    Pi = np.zeros((n, NB))
    for k in (0, 1):
        idx = np.nonzero(S == k)[0]; j = np.floor(L[idx]/LS).astype(int); u = L[idx]/LS - j
        for jj in np.unique(j):
            m = j == jj; q = idx[m]; Pi[q] = (1 - u[m])[:, None]*node(k, int(jj))[None, :] + u[m][:, None]*node(k, int(jj) + 1)[None, :]
    hm = HH['B10'].reshape(n, NB, 2).sum(2).astype(float); Dv = hm/XB - Pi; seP = np.sqrt(NP*XB*Pi.mean(0))/(NP*XB); Y = (Dv/seP) @ CM[1:8].T; yb = Y.mean(0); s2 = Y.var(0, ddof=1)
    e0 = {'l1': float((yb**2 + s2/NP*(1 - NP/n)).sum()/80)}; Mb = n//3; Z = Y[order[:3*Mb]].reshape(Mb, 3, 7).mean(1); m3 = round(NP/3)
    e0['l3'] = float((m3*3/NP)*(Z.mean(0)**2 + Z.var(0, ddof=1)/m3*(1 - m3/Mb)).sum()/80); Cy = np.vstack([np.zeros(7), np.cumsum(Y[order], 0)]); e0['stretch'] = float((((Cy[NP:] - Cy[:-NP])/NP)**2).sum(1).mean()/80)
    rng11 = np.random.default_rng(20261105); ob, pe, p2 = [], [], []
    for _ in range(1000):
        ii = np.sort(rng11.permutation(n)[:NP]); H = hm[ii].sum(0); dm = Dv[ii].mean(0); se_o = np.sqrt(np.maximum(H, 1))/(NP*XB); se_p = np.sqrt(NP*XB*Pi[ii].mean(0))/(NP*XB); se_2 = se_o.copy(); se_2[:2] = se_p[:2]
        f_ = lambda e: float(np.sum((CM[1:8] @ (dm/e))**2)/80); ob.append(f_(se_o)); pe.append(f_(se_p)); p2.append(f_(se_2))
    ob, pe, p2 = map(np.array, (ob, pe, p2)); ms = lambda a: (float(a.mean()), float(a.std(ddof=1)/math.sqrt(len(a))))
    dpe, dop, d2 = ms(pe - e0['l1']), ms(ob - pe), ms(p2 - pe); Rc = np.corrcoef(Y.T)
    SQ30 = {p: np.array([0] + [1 if r in set((x*x) % p for x in range(1, p)) else -1 for r in range(1, p)], float) for p in (3, 5, 7, 11, 13, 17, 19, 23, 29)}
    def leg(p): r8 = Ds % 8; return np.where(Ds % 2 == 0, 0, np.where((r8 == 1) | (r8 == 7), 1, -1)).astype(float) if p == 2 else SQ30[p][Ds % p]
    ch = np.stack([leg(p) for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29)], 1); Xr = np.hstack([np.ones((n, 1)), ch, ch*ch]); cf = np.linalg.lstsq(Xr, Y, rcond=None)[0]
    reg30 = float((yb**2 + (Y - Xr @ cf).var(0, ddof=1)/NP*(1 - NP/n)).sum()/80)
    rel = max(abs(e0['l1']/RL['designs']['l1']['layer0'] - 1), abs(e0['l3']/RL['designs']['l3']['layer0'] - 1), abs(e0['stretch']/RL['stretch']['layer0'] - 1), abs(reg30/RL['regression']['30']['layer0'] - 1))
    ok11 = rel < 0.03 and abs(dpe[0]) < 3*dpe[1] + 2e-4 and dop[0] > 3*dop[1] and abs(d2[0]) < 0.35*dop[0] and Rc[0, 6] > 0.4 and abs(Rc[0, 6] - RL['coupling']['pairs']['1-7']) < 0.02 and e0['l3'] < e0['l1'] and e0['stretch'] < 0.3*e0['l1']
    check('exact expectations (own y_ik in B10): layer 0 of random sample, triples, stretch and regression p <= 30 = reference within 3 %; own Pearson draws = layer 0; observed-count statistic above, carried by the two lowest bins; coupling of modes 1-7', ok11,
          'layer 0 random %.4f / triples %.4f / stretch %.4f / regression p<=30 %.4f (reference %.4f / %.4f / %.4f / %.4f, largest relative difference %.3f); Pearson - layer 0 %+.5f +- %.5f; observed - Pearson %+.4f +- %.4f, with Pearson in the two lowest bins %+.4f +- %.4f; corr(1,7) %.3f (reference %.3f)'
          % (e0['l1'], e0['l3'], e0['stretch'], reg30, RL['designs']['l1']['layer0'], RL['designs']['l3']['layer0'], RL['stretch']['layer0'], RL['regression']['30']['layer0'], rel, dpe[0], dpe[1], dop[0], dop[1], d2[0], d2[1], Rc[0, 6], RL['coupling']['pairs']['1-7']))
# ---- 12) section 4.6: intermodulation - chi_D(n) of composite n (products of chi_D(p)) carry the modes 8 - k
if len(sys.argv) > 10:
    RI = json.load(open(sys.argv[9]))['families']['B10']; P29 = (2, 3, 5, 7, 11, 13, 17, 19, 23, 29); LG = {p: leg(p) for p in P29}
    CMP = []
    for m in range(6, 55):
        fs = [p for p in P29 if m % p == 0]
        if math.log(m) < 4 and len(fs) >= 2 and math.prod(fs) == m: CMP.append(m)
    Cm_ = np.stack([np.prod([LG[p] for p in P29 if m % p == 0], 0) for m in CMP], 1); Xb = np.hstack([np.ones((n, 1))] + [np.stack([LG[p] for p in P29], 1), np.stack([LG[p] for p in P29], 1)**2])
    def r2_(X_): cf_ = np.linalg.lstsq(X_, Y, rcond=None)[0]; return 1 - (Y - X_ @ cf_).var(0, ddof=1)/s2
    g0 = r2_(Xb); gain = r2_(np.hstack([Xb, Cm_])) - g0; gnull = r2_(np.hstack([Xb, Cm_[np.random.default_rng(20261106).permutation(n)]])) - g0
    ratio = float(gain[5:7].mean()/gain[0:4].mean())
    check('intermodulation (own y_ik and Legendre symbols in B10): chi_D(n) of %d composite n < e^4 raise the modes 6-7 many times more than 1-4; permuted features give almost nothing' % len(CMP),
          len(CMP) == 16 and ratio > 5 and gain[6] > 0.12 and abs(gain[6] - RI['windows'][0]['gain'][6]) < 0.05 and float(np.max(gnull)) < 0.002,
          'gain k = 1 ... 7: %s; ratio 6-7 / 1-4 %.1f (reference with p <= 300: %.1f); permuted at most %.4f' % (' '.join('%.3f' % x for x in gain), ratio, RI['ratio_modes_67_to_14'], float(np.max(gnull))))
    # ---- 13) section 4.5: crossing of the first zero of even characters with USp(2N_eff), own route
    RX = json.load(open(sys.argv[10]))['families']; dmax = 0; info = []; okx = True
    for f in ('C100', 'B10'):
        Ds = np.array(DATA[f][0]); S = (Ds < 0).astype(int); xg = np.arange(0, 2001)/1000; Fm = {}; Fd = {}
        for sign, k in (('plus', 0), ('minus', 1)):
            x1 = np.sort(X1[f][S == k]); x1 = x1[np.isfinite(x1)]; Fd[sign] = np.searchsorted(x1, xg + 1e-12, side='right')/len(x1)
            Fm[sign] = np.array([1 - E0(v, NEF[f + '|' + sign]) for v in xg])
        dp = Fd['plus'] - Fm['plus']; dmn = Fd['minus'] - Fm['minus']; dd = dp - dmn
        def root(y, a, b, sg=None):
            for i in range(int(round(a*1000)), int(round(b*1000))):
                if y[i] != 0 and y[i]*y[i + 1] < 0 and (sg is None or np.sign(y[i + 1]) == sg): return i/1000 - y[i]*0.001/(y[i + 1] - y[i])
            return None
        xc = root(dp, 0.3, 0.8, -1); xe = root(dd, 0.3, 0.8); imn = int(np.argmin(dp[int(xc*1000):2000])) + int(xc*1000); ks = 1.36/math.sqrt(int(np.sum(S == 0)))
        jr = np.nonzero(dp[imn:] > -ks)[0]; xr = float(xg[imn + jr[0]]) if len(jr) else None; q = RX[f]['even']
        Fs = np.interp(xg - q['null_models']['shift']['delta'], xg, Fm['plus'], left=0.0); sh = root(Fs - Fm['plus'], 0.05, 1.99)
        dmax = max(dmax, abs(xc - q['crossing']), abs(xe - q['crossing_even_odd']), abs(xr - q['reenter_ks_band'])); okx = okx and sh is None and abs(xc - 0.5) < 2*q['crossing_se']
        info.append('%s crossing %.4f (reference %.4f +- %.4f), even = odd %.4f, minimum at %.3f, back in the KS band %.3f' % (f, xc, q['crossing'], q['crossing_se'], xe, xg[imn], xr))
        if f == 'C100':
            m = S == 0; pm0 = pred_mean(LL[f][m], S[m]); kal = []; KR = {round(c['a'], 3): c['crossing'] for c in RX[f]['calibration_test']['even']}
            for a in (0.4, 0.6):
                Na = n_eff(float(np.mean(pm0[:int(round(a/XB))])), a); ga = np.arange(int(round((a - 0.2)*1000)), int(round((a + 0.25)*1000)) + 1)
                ya = np.full(len(xg), np.nan); ya[ga] = dp[ga] + Fm['plus'][ga] - np.array([1 - E0(xg[i], Na) for i in ga]); xa = root(np.nan_to_num(ya), a - 0.2, a + 0.25, -1)
                kal.append((a, Na, xa)); dmax = max(dmax, abs(xa - KR[a])); okx = okx and abs(xa - a) < 0.08
            okx = okx and (kal[1][2] - kal[0][2])/0.2 > 0.8
            info.append('calibration on [0, a] in C100: ' + ', '.join('a = %.1f: N_eff %.3f, crossing %.4f (reference %.4f)' % (a, Na, xa, KR[a]) for a, Na, xa in kal) + ', slope %.2f' % ((kal[1][2] - kal[0][2])/0.2))
    check('crossing of the first zero (own first zeros, N_eff and Fredholm GL 60 on a grid of 0.001): = reference within 2e-3, within 2 errors of 1/2 in C100 and B10; a shifted model gives no crossing; with N_eff fixed on [0, a] the crossing follows a (a = 0.4, 0.6)', okx and dmax < 2e-3,
          '; '.join(info) + '; largest difference %.1e' % dmax)
# ---- 14) section 4.7: layer 0 of stratifications exactly (own y_ik of check 11), the line through two points with the values by layers (own W100 of check 9)
if len(sys.argv) > 13:
    RR = json.load(open(sys.argv[11])); DsB = np.array(DATA['B10'][0], np.int64); nB = len(DsB); yB = Y.mean(0); l0 = {}
    for M in (120, 840):
        inv, cnt, k = strat_alloc((DsB < 0).astype(np.int64)*M + DsB % M); wc = k/NP; mu_c = np.stack([Y[inv == c].mean(0) for c in range(len(cnt))])
        s2_c = np.stack([Y[inv == c].var(0, ddof=1) if cnt[c] > 1 else np.zeros(7) for c in range(len(cnt))])
        ev = (wc[:, None]*mu_c).sum(0); vv = (wc[:, None]**2*s2_c/np.maximum(k, 1)[:, None]*(1 - k/cnt)[:, None]).sum(0); l0['mod%d' % M] = float((ev**2 + vv).sum()/80)
    l0['random'] = float((yB**2 + Y.var(0, ddof=1)/NP*(1 - NP/nB)).sum()/80); RB = RR['families']['B10']['designs']
    rel14 = max(abs(l0[d]/RB[d]['layer0']['low'] - 1) for d in l0); chi2l = 0.0; npl = 0; posl = 0
    for f in ('B10', 'C100', 'D100'):
        F = RR['families'][f]; Dd = F['designs']; R0, Rse = Dd['random']['estimate']['low']; S0, Sse = F['bands']['own']['low'][:2]; Ws = own[f]['consecutive']
        q = np.array([(1 - own[f]['mod%d' % M])/(1 - Ws) for M in STRAT]); m = np.array([Dd['mod%d' % M]['estimate']['low'][0] for M in STRAT]); se = np.array([Dd['mod%d' % M]['estimate']['low'][1] for M in STRAT])
        d = m - (R0*(1 - q) + S0*q); Cm = np.diag(se**2) + Rse**2*np.outer(1 - q, 1 - q) + Sse**2*np.outer(q, q); chi2l += float(d @ np.linalg.solve(Cm, d)); npl += len(STRAT); posl += int(np.sum(d > 0))
    zc = [d['layer0_check_z'][s_] for f in ('B10', 'C100', 'D100') for d in RR['families'][f]['designs'].values() for s_ in ('low', 'high', 'chi05', 'chi01')]; mz2 = float(np.mean(np.square(zc)))
    check('recomputation by layers (own y_ik in B10, own allocation and exact stratified variance): layer 0 of random, mod 120 and mod 840 = reference within 3 %; the line through two points with the values by layers and own W100 is rejected (chi2 > 400 for 60) = reference; layer-0 check of the reference draws: mean z^2 < 1.3',
          rel14 < 0.03 and npl == 60 and chi2l > 400 and abs(chi2l - RR['line']['layered_bands']['chi2']) < 0.5 and mz2 < 1.3,
          'layer 0 random %.4f / mod 120 %.4f / mod 840 %.4f (reference %.4f / %.4f / %.4f, largest relative difference %.3f); line chi2 %.1f / %d (reference %.1f), %d positive; mean z^2 %.2f of %d'
          % (l0['random'], l0['mod120'], l0['mod840'], RB['random']['layer0']['low'], RB['mod120']['layer0']['low'], RB['mod840']['layer0']['low'], rel14, chi2l, npl, RR['line']['layered_bands']['chi2'], posl, mz2, len(zc)))
    # ---- 15) section 4.7: two-level part of the first zero (own eigenvalues of the kernel, GL 60) and the calibration of N_eff on [0, a] (own route of check 6)
    RT = json.load(open(sys.argv[12]))['families']; info = []; ok15 = True
    def counts(a, N):
        x = 0.5*a*(X60 + 1); w = 0.5*a*W60; Sn = lambda z: np.where(np.abs(z) < 1e-12, (2*N + 1)/(2*N), np.sin((2*N + 1)*np.pi*z/(2*N))/(2*N*np.sin(np.pi*z/(2*N)) + 1e-300))
        lam = np.clip(np.linalg.eigvalsh(np.sqrt(w)[:, None]*(Sn(x[:, None] - x[None, :]) - Sn(x[:, None] + x[None, :]))*np.sqrt(w)[None, :]), 0, 1); return float(lam.sum()), float(lam.sum()**2 - (lam**2).sum())
    Ds = np.array(DATA['C100'][0], np.int64); S = (Ds < 0).astype(int); G = DATA['C100'][1]; cal = {}
    for sign, k in (('plus', 0), ('minus', 1)):
        m = np.nonzero(S == k)[0]; n2 = 0
        for i in m:
            g = G[i]
            if X1['C100'][i] <= 0.5 and len(g) > 1 and th(int(Ds[i]), g[1])/math.pi <= 0.5: n2 += 1
        en, enn = counts(0.5, NEF['C100|' + sign]); dat = 2*n2/len(m); ref = RT['C100'][sign]['rows']['0.5']
        ok15 = ok15 and enn < 5e-4 and dat < 3e-4 and abs(enn - ref['model']['ENN1']) < 2e-5 and abs(en - ref['model']['EN']) < 1e-3
        pm0 = pred_mean(LL['C100'][m], S[m]); x1 = X1['C100'][m]; x1 = x1[np.isfinite(x1)]; cal[sign] = []
        for a in (0.3, 0.7):
            Na = n_eff(float(np.mean(pm0[:int(round(a/XB))])), a); mf = quad(lambda s_: E0(s_, Na), 0, min(3.0, 0.98*Na), limit=200, epsabs=1e-7)[0]
            rc = [c for c in RT['C100'][sign]['calibration'] if abs(c['a'] - a) < 1e-9][0]; cal[sign].append((a, Na, float(x1.mean() - mf), rc['difference'])); ok15 = ok15 and abs(x1.mean() - mf - rc['difference']) < 5e-4
        info.append('%s: E[N(N-1)] at 1/2 model %.5f (reference %.5f), data %.5f; ' % (sign, enn, ref['model']['ENN1'], dat) + ', '.join('a = %.1f: N_eff %.3f, mean x1 - model %+.4f (reference %+.4f)' % c for c in cal[sign]))
    se1 = float(np.std(X1['C100'][S == 0][np.isfinite(X1['C100'][S == 0])])/math.sqrt(int(np.sum(S == 0))))
    ok15 = ok15 and (cal['plus'][0][2] - cal['plus'][1][2]) > 10*se1 and max(abs(c[2]) for c in cal['minus']) < 0.003
    check('first zero (C100, own route): the two-level part E[N(N-1)] at a = 1/2 below 5e-4 in USp(2N_eff) and in the data; with N_eff fixed on [0, a] the mean shift of even characters falls by more than 10 errors from a = 0.3 to 0.7, odd characters within 0.003 = reference_two_level.json', ok15, '; '.join(info))
    # ---- 16) section 4.7: completeness - the close pair of zeros missed by the step spacing / 8, own Hardy function on a fine grid
    MZ = json.load(open(sys.argv[13])); ex = MZ['examples'][0]; Dm = int(ex['D']); Ds = np.array(DATA['B10'][0], np.int64); im = int(np.nonzero(Ds == Dm)[0][0]); gm = DATA['B10'][1][im]
    a_, b_ = min(ex['t_missing']) - 0.03, max(ex['t_missing']) + 0.03; Lm = pari.lfuninit(pari.lfuncreate(Dm), pari('[1/2, 0, %.6f]' % (b_ + 0.2))); ts = np.arange(a_, b_, 0.0005)
    hv = [float(pari.lfunhardy(Lm, float(t))) for t in ts]; sc_ = [float(ts[i] + 0.00025) for i in range(len(ts) - 1) if hv[i]*hv[i + 1] < 0]; stored = [v for v in gm if a_ <= v <= b_]
    ok16 = len(sc_) == 2 and max(abs(u - v) for u, v in zip(sorted(sc_), sorted(ex['t_missing']))) < 1e-3 and len(stored) == 0 and MZ['missing_L']/MZ['sample'] < 0.002 and MZ['same_as_data'] == MZ['sample']
    check('completeness (own Hardy function, step 0.0005): the close pair of zeros of the reference example is there and missing in the data; at most 1 L in 500 of the resampled sample misses zeros', ok16,
          'D = %d: sign changes at %s (reference %s), stored zeros in [%.3f, %.3f]: %d; sample %d L: %d with missed zeros' % (Dm, ['%.4f' % v for v in sc_], ex['t_missing'], a_, b_, len(stored), MZ['sample'], MZ['missing_L']))
if len(sys.argv) > 14:
    # ---- 17) section 4.8: the family E100 (E10 and the new blocks): own composition, own unfolding and first zero against reference_families_E100.json
    RE = json.load(open(sys.argv[14]))['families']['E100']; dE, gE, _ = load('E100'); DsE = np.array(list(DATA['E10'][0]) + list(dE), np.int64); GE = list(DATA['E10'][1]) + list(gE)
    own = fundamental_range(10**10, 10**10 + 2*10**5); comp17 = sorted(own) == sorted(DsE.tolist()) and len(set(DsE.tolist())) == len(DsE)
    x1E = np.array([th(int(D), g[0])/math.pi if len(g) else np.nan for D, g in zip(DsE, GE)]); x1E[~(x1E < 4)] = np.nan
    LE = np.log(np.abs(DsE.astype(float))/math.pi); SE = (DsE < 0).astype(int); dv = {}; dd17 = 0; dm17 = 0
    for sign, k in (('plus', 0), ('minus', 1)):
        m = SE == k; d0p = float(np.mean(pred_mean(LE[m], SE[m])[:10])); Ne = n_eff(d0p); mef = quad(lambda s_: E0(s_, Ne), 0, min(3.0, 0.98*Ne), limit=200, epsabs=1e-7)[0]
        x = x1E[m]; x = x[np.isfinite(x)]; dv[sign] = (float(x.mean() - mef), float(x.std()/math.sqrt(len(x))), int(m.sum()))
        dd17 = max(dd17, abs(float(x.mean()) - RE[sign]['x1_mean'])); dm17 = max(dm17, abs(mef - RE[sign]['x1_usp2n_eff']))
    lE = float(LE.mean()); zl = (dv['plus'][0] - 0.0938/lE)/dv['plus'][1]
    ok17 = comp17 and dd17 < 1e-5 and dm17 < 3e-3 and abs(dv['minus'][0]/dv['minus'][1]) < 2 and zl < -1.5
    check('E100 (Section 4.8; own sieve, own unfolding, own N_eff and Fredholm GL 60): the discriminants with 10^10 < |D| < 10^10 + 2*10^5 = the files; mean first zero and model = reference; odd within 2 errors; even below 0.094/l', ok17,
          'composition %s (%d L, %d even); data means within %.1e, model within %.1e; even %+.4f +- %.4f (z against 0.0938/l %.1f), odd %+.4f +- %.4f' % (comp17, len(DsE), dv['plus'][2], dd17, dm17, dv['plus'][0], dv['plus'][1], zl, dv['minus'][0], dv['minus'][1]))
print('%d of %d checks passed, %.0f s' % (sum(OK), len(OK), time.time() - t0)); sys.exit(0 if all(OK) else 1)
