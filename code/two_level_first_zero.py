#!/usr/bin/env python3
"""
Low-lying zeros of quadratic Dirichlet L-functions, study III, Section 4.7: the first zero of even and odd characters split into its one-level and
two-level parts, and the even-odd difference of the mean first zero for every calibration interval of N_eff.

usage:  python3 -I two_level_first_zero.py ZEROS_ROOT REFERENCE_FAMILIES_JSON OUT_DIR      (about a minute; writes reference_two_level.json)
        python3 -I two_level_first_zero.py ZEROS_ROOT reference_families_E100.json OUT_DIR --families E100   (Section 4.8; writes reference_two_level_E100.json)

For the number N(a) of zeros in [0, a] of the unfolded variable, P(x1 <= a) = E[N(a)] - sum_{k>=2} P(N(a) >= k): the first term is fixed by the
one-level density, the rest needs the two-level density.  In USp(2N) the zeros in [0, a] form a determinantal process with the kernel
S_N(x - y) - S_N(x + y); with the eigenvalues lambda_j of the kernel on [0, a] (Gauss-Legendre, 60 nodes) N(a) is a sum of independent Bernoulli
variables, so E[N], E[N(N-1)] = (sum lambda)^2 - sum lambda^2 and P(N >= k) follow exactly.  Data and USp(2N_eff) (N_eff of the family and sign,
fixed by the ratios conjecture in [0, 1/2]) are compared for a = 0.3 ... 1.0.  Second, N_eff(a) is fixed by the density of the ratios conjecture in
[0, a] for a = 0.3 ... 0.7 separately for both signs, and the mean first zero of the data is compared with USp(2N_eff(a)).
"""
import sys, os, json, math, time, numpy as np, scipy.special as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
t0 = time.time(); ROOT, REF, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT_DIR, exist_ok=True)
ONLY = sys.argv[sys.argv.index('--families') + 1].split(',') if '--families' in sys.argv else None
FAMS = ('B10', 'C100', 'R10', 'D100', 'E10') if not ONLY else tuple(ONLY); AS = (0.3, 0.4, 0.5, 0.6, 0.7, 1.0); CAL_A = (0.3, 0.4, 0.5, 0.6, 0.7)
J = json.load(open(REF))
def theta(d, t): k = 0 if d > 0 else 1; return 0.5*np.asarray(t)*math.log(abs(d)/math.pi) + np.imag(sp.loggamma(0.25 + 0.5*k + 0.5j*np.asarray(t)))
XG, WG = np.polynomial.legendre.leggauss(60)
def S_(z, N): return np.where(np.abs(z) < 1e-12, (2*N + 1)/(2*N), np.sin((2*N + 1)*np.pi*z/(2*N))/(2*N*np.sin(np.pi*z/(2*N)) + 1e-300))
def eig(a, N):
    x = 0.5*a*(XG + 1); w = 0.5*a*WG; K = S_(x[:, None] - x[None, :], N) - S_(x[:, None] + x[None, :], N)
    return np.clip(np.linalg.eigvalsh(np.sqrt(w)[:, None]*K*np.sqrt(w)[None, :]), 0, 1)
def counting(a, N, kmax=6):
    """E[N], E[N(N-1)] and P(N >= k), k = 1 ... kmax, of USp(2N) in [0, a] (Poisson-binomial distribution of the eigenvalues)"""
    lam = eig(a, N); p = np.zeros(len(lam) + 1); p[0] = 1.0
    for l_ in lam: p[1:] = p[1:]*(1 - l_) + p[:-1]*l_; p[0] *= (1 - l_)
    ge = [float(p[k:].sum()) for k in range(1, kmax + 1)]
    return float(lam.sum()), float(lam.sum()**2 - (lam**2).sum()), ge
def E0(s_, N):
    if s_ <= 0: return 1.0
    return float(np.prod(1 - eig(s_, N)))
XM, WM = np.polynomial.legendre.leggauss(120)
def mean_first(N):
    b = min(3.0, 0.98*N); x = 0.5*b*(XM + 1); w = 0.5*b*WM; return float(sum(wi*E0(xi, N) for xi, wi in zip(x, w)))
def W_N(x, N): y = np.pi*x/N; return 1 + 1/(2*N) - np.sin((2*N + 1)*y)/(2*N*np.sin(np.maximum(y, 1e-12)))
def dens(N, a): xs = np.linspace(1e-9, a, 4001); return float(np.trapezoid(W_N(xs, N), xs)/a)
def neff_a(d, a):
    lo, hi = 0.55, 80.0
    for _ in range(60):
        mid = 0.5*(lo + hi)
        if dens(mid, a) > d: lo = mid
        else: hi = mid
    return 0.5*(lo + hi)
KSG = np.round(np.arange(0.005, 3.0, 0.005), 4)
OUT = dict(study='III', section='4.7', a=AS, calibration_a=CAL_A, families={})
for f in FAMS:
    Xz = {0: [], 1: []}; X1 = {0: [], 1: []}
    for D, g in RZ.study_zeros(f, ROOT):
        x = theta(int(D), np.array(g))/math.pi; Xz[0 if D > 0 else 1].append(x[x < 1.5])
        if len(x) and x[0] < 4: X1[0 if D > 0 else 1].append(float(x[0]))
    R = {}
    for k, sg in ((0, 'plus'), (1, 'minus')):
        Q = J['families'][f][sg]; Ne = float(Q['N_eff']); n = len(Xz[k]); rows = {}
        for a in AS:
            c = np.array([np.sum(x <= a) for x in Xz[k]], float); m1, m2, ge = counting(a, Ne)
            d = dict(n=n, EN=[float(c.mean()), float(c.std(ddof=1)/math.sqrt(n))], ENN1=[float((c*(c - 1)).mean()), float((c*(c - 1)).std(ddof=1)/math.sqrt(n))],
                     F=[float(np.mean(c >= 1)), float(math.sqrt(np.mean(c >= 1)*(1 - np.mean(c >= 1))/n))], P2=[float(np.mean(c >= 2)), float(math.sqrt(np.mean(c >= 2)*(1 - np.mean(c >= 2))/n))],
                     T2=float(c.mean() - np.mean(c >= 1)), model=dict(EN=m1, ENN1=m2, F=ge[0], P2=ge[1], T2=m1 - ge[0]))
            d['dEN'] = d['EN'][0] - m1; d['dT2'] = d['T2'] - d['model']['T2']; d['dF'] = d['F'][0] - ge[0]; d['dENN1'] = d['ENN1'][0] - m2
            rows[str(a)] = d
        pr = np.array(Q['prediction_0025']); cal = []; x1s = np.sort(np.array(X1[k]))
        for a in CAL_A:
            k_ = int(round(a/0.025)); Ne_ = neff_a(float(pr[:k_].mean()), a); mf = mean_first(Ne_)
            Fm = np.array([1 - E0(v, Ne_) if v < 0.98*Ne_ else 1.0 for v in KSG]); Fd = np.searchsorted(x1s, KSG, side='right')/len(x1s); ks = float(np.max(np.abs(Fd - Fm)))
            cal.append(dict(a=a, N_eff=Ne_, x1_model=mf, x1_data=float(Q['x1_mean']), x1_se=float(Q['x1_se']), difference=float(Q['x1_mean']) - mf, ks=ks, ks_critical=1.36/math.sqrt(len(x1s))))
        R[sg] = dict(n=n, N_eff=Ne, rows=rows, calibration=cal, x1_model_reference=float(Q['x1_usp2n_eff']), x1_model_check=mean_first(Ne))
        print('%-5s %-5s N_eff %.3f: a = 0.5 E[N] %.4f / %.4f, E[N(N-1)] %.4f / %.4f, F %.4f / %.4f (data / model); dF = dE[N] %+.4f - dT2 %+.4f; x1 mean - model by a: %s' % (
              f, sg, Ne, rows['0.5']['EN'][0], rows['0.5']['model']['EN'], rows['0.5']['ENN1'][0], rows['0.5']['model']['ENN1'], rows['0.5']['F'][0], rows['0.5']['model']['F'],
              rows['0.5']['dEN'], rows['0.5']['dT2'], ' '.join('%.1f:%+.4f (KS %.4f/%.4f)' % (q['a'], q['difference'], q['ks'], q['ks_critical']) for q in cal)), round(time.time() - t0), 's', flush=True)
    R['even_minus_odd_by_calibration'] = [dict(a=a, difference=R['plus']['calibration'][i]['difference'] - R['minus']['calibration'][i]['difference'],
                                               se=math.hypot(R['plus']['calibration'][i]['x1_se'], R['minus']['calibration'][i]['x1_se'])) for i, a in enumerate(CAL_A)]
    OUT['families'][f] = R
json.dump(OUT, open(os.path.join(OUT_DIR, 'reference_two_level%s.json' % ('_' + '_'.join(ONLY) if ONLY else '')), 'w')); print('done', round(time.time() - t0), 's', flush=True)
