#!/usr/bin/env python3
"""
Low-lying zeros of quadratic Dirichlet L-functions, study III: where the distribution of the first zero of even characters crosses
USp(2N_eff) (Section 4.5, Figure 7 left).

usage:  python3 -I first_zero_crossing.py ZEROS_ROOT REFERENCE_FAMILIES_JSON OUT_DIR     (about 20 seconds; writes reference_first_zero_crossing.json)

For each family and sign the first unfolded zero x1 = theta_D(gamma_1)/pi (as compute_families.py, x1 below 4) is read, and its distribution
function is compared with USp(2N_eff) (Fredholm determinant, Gauss-Legendre 40, N_eff of the family and sign from reference_families.json),
on a grid of 0.001 instead of 0.02.  For dF(x) = F_data(x) - F_USp(2N_eff)(x):
  crossing      the change of sign of dF between 0.3 and 0.8 (linear interpolation), and the crossing of the even and the odd curve dF+ = dF-;
  extremes      position and value of the maximum before and of the minimum after the crossing;
  return        where dF of the even characters re-enters the band of the Kolmogorov-Smirnov critical value 1.36/sqrt(n) after the minimum,
                where it comes within one pointwise error sqrt(F(1 - F)/n) of zero for good, its value at x = 1.5 and any further change of sign;
  errors        bootstrap over the L-functions (400 resamples, multinomial on histograms of 0.001); the model is fixed;
  null models   a pure shift of the model by the measured difference of the mean first zero, and a pure change of scale with the same mean:
                both give dF of one sign only, i.e. no crossing at all.
Calibration test (the decisive null model): N_eff is fixed by the density of the ratios conjecture in [0, 1/2], and for small x the distribution function of
the first zero is almost the mean number of zeros below x; the crossing may therefore be set by the calibration.  N_eff(a) is fixed by the density in [0, a],
a = 0.3 ... 0.7 (prediction of the family from reference_families.json, trapezoidal rule on W_N), and the crossing of the data with USp(2N_eff(a)) is located;
a straight line crossing = c0 + c1 a is fitted (C100 and B10, where the crossing is measurable).
The models end at 0.98 N_eff (USp(2N): x1 <= N).  The counts of x1 on the grid (bin j: x_(j-1) < x1 <= x_j, last bin x1 above 3) are written as well.
"""
import sys, os, json, math, time, numpy as np, scipy.special as sp
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
t0 = time.time(); ROOT, REF, OUT_DIR = sys.argv[1], sys.argv[2], sys.argv[3]; os.makedirs(OUT_DIR, exist_ok=True)
FAMS = ('A', 'B10', 'C100', 'R10', 'D100', 'E10'); DX = 0.001; XG_ = np.round(np.arange(0, 3.0 + 1e-9, DX), 4); BOOT = 400; SEED = 346811
J = json.load(open(REF))
def theta(d, t): k = 0 if d > 0 else 1; return 0.5*t*math.log(abs(d)/math.pi) + float(np.imag(sp.loggamma(0.25 + 0.5*k + 0.5j*t)))
XG, WG = np.polynomial.legendre.leggauss(40)
def E0(s_, N):                                                     # as compute_families.py
    if s_ <= 0: return 1.0
    x = 0.5*s_*(XG + 1); w = 0.5*s_*WG
    S_ = lambda z: np.where(np.abs(z) < 1e-12, (2*N + 1)/(2*N), np.sin((2*N + 1)*np.pi*z/(2*N))/(2*N*np.sin(np.pi*z/(2*N)) + 1e-300))
    K = S_(x[:, None] - x[None, :]) - S_(x[:, None] + x[None, :]); A = np.eye(len(x)) - np.sqrt(w)[:, None]*K*np.sqrt(w)[None, :]
    return float(np.linalg.det(A))
def model(N): return 1 - np.array([E0(s_, N) if s_ < 0.98*N else 0.0 for s_ in XG_])
def W_N(x, N): y = np.pi*x/N; return 1 + 1/(2*N) - np.sin((2*N + 1)*y)/(2*N*np.sin(np.maximum(y, 1e-12)))
def dens(N, a): xs = np.linspace(1e-9, a, 4001); return float(np.trapezoid(W_N(xs, N), xs)/a)
def neff_a(d, a):
    lo, hi = 0.55, 80.0
    for _ in range(60):
        mid = 0.5*(lo + hi)
        if dens(mid, a) > d: lo = mid
        else: hi = mid
    return 0.5*(lo + hi)
CAL_A = (0.3, 0.4, 0.5, 0.6, 0.7)
def roots(y, a, b, sgn=None):
    """changes of sign of y on the grid between a and b (linear interpolation); sgn = -1 only from + to -, +1 only from - to +"""
    i0, i1 = int(round(a/DX)), int(round(b/DX)); o = []
    for i in range(i0, i1):
        if y[i] == 0: continue
        if y[i]*y[i + 1] < 0 and (sgn is None or np.sign(y[i + 1]) == sgn): o.append(float(XG_[i] - y[i]*DX/(y[i + 1] - y[i])))
    return o
OUT = dict(study='III', grid=DX, bootstrap=BOOT, seed=SEED, families={})
rng = np.random.default_rng(SEED)
for f in FAMS:
    X1 = {0: [], 1: []}
    for D, g in RZ.zeros(f, ROOT):
        if not len(g): continue
        x = theta(int(D), float(g[0]))/math.pi
        if x < 4: X1[0 if D > 0 else 1].append(x)
    R = {}; Fd = {}; Fm = {}; cnt = {}; nn = {}
    for k, sg in ((0, 'plus'), (1, 'minus')):
        x1 = np.sort(np.array(X1[k])); n = len(x1); Q = J['families'][f][sg]; Ne = float(Q['N_eff'])
        assert n == Q['x1_n'], (f, sg, n, Q['x1_n'])
        Fmod = model(Ne); chk = float(np.max(np.abs(Fmod[::20][:len(Q['x1_cdf_usp2n_eff'])] - np.array(Q['x1_cdf_usp2n_eff'][:len(Fmod[::20])]))))
        c = np.histogram(x1, bins=np.concatenate([[-1.0], XG_ + 1e-12, [9.0]]))[0]                       # c_0: x1 <= 0 (none); c_j: x_(j-1) < x1 <= x_j; last: x1 above 3
        F = np.cumsum(c)[:len(XG_)]/n; Fd[k] = F; Fm[k] = Fmod; cnt[k] = c; nn[k] = n
        R[sg] = dict(n=n, N_eff=Ne, model_check_against_reference=chk, ks_critical=1.36/math.sqrt(n), counts_grid=c.tolist(), x1_mean=float(x1.mean()), x1_mean_model=float(Q['x1_usp2n_eff']))
    dF = {k: Fd[k] - Fm[k] for k in (0, 1)}; sig = {k: np.sqrt(np.maximum(Fd[k]*(1 - Fd[k]), 1e-300)/nn[k]) for k in (0, 1)}
    def describe(dp, dm):
        o = {}; r_ = roots(dp, 0.3, 0.8, -1); o['crossing'] = r_[0] if r_ else None; r2 = roots(dp - dm, 0.3, 0.8); o['crossing_even_odd'] = r2[0] if r2 else None
        if o['crossing'] is not None:
            ic = int(o['crossing']/DX); imx = int(np.argmax(dp[:ic + 1])); imn = ic + int(np.argmin(dp[ic:int(2.0/DX)])); o.update(max_x=float(XG_[imx]), max=float(dp[imx]), min_x=float(XG_[imn]), min=float(dp[imn]))
            o['further_crossings'] = roots(dp, XG_[imn], 2.999); r3 = roots(dp - dm, XG_[imn], EDGE); o['second_crossing_even_odd'] = r3[0] if r3 else None
        return o
    EDGE = min(2.998, float(np.floor(0.98*min(R['plus']['N_eff'], R['minus']['N_eff'])/DX)*DX - DX))        # end of the support of the models (E0 set to 0 from 0.98 N_eff on)
    B = {'crossing': [], 'crossing_even_odd': [], 'second_crossing_even_odd': [], 'max_x': [], 'min_x': [], 'reenter': []}
    ks0 = R['plus']['ks_critical']
    def reenter(dp, imn):
        j = np.nonzero(dp[imn:] > -ks0)[0]; return float(XG_[imn + j[0]]) if len(j) else None
    d0 = describe(dF[0], dF[1]); imn0 = int(round(d0['min_x']/DX)) if d0.get('min_x') is not None else None
    for b in range(BOOT):
        cb = {k: rng.multinomial(nn[k], cnt[k]/nn[k]) for k in (0, 1)}; db = {k: np.cumsum(cb[k])[:len(XG_)]/nn[k] - Fm[k] for k in (0, 1)}; o = describe(db[0], db[1])
        for key in ('crossing', 'crossing_even_odd', 'second_crossing_even_odd', 'max_x', 'min_x'):
            if o.get(key) is not None: B[key].append(o[key])
        if o.get('min_x') is not None:
            v = reenter(db[0], int(round(o['min_x']/DX)))
            if v is not None: B['reenter'].append(v)
    se = {k: (float(np.std(v, ddof=1)) if len(v) > 1 else None) for k, v in B.items()}
    # return of the even curve
    ret = {}
    if imn0 is not None:
        ret['reenter_ks_band'] = reenter(dF[0], imn0); j = np.nonzero(np.abs(dF[0][imn0:]) > sig[0][imn0:])[0]
        ret['within_one_error_from'] = float(XG_[imn0 + j[-1] + 1]) if len(j) and imn0 + j[-1] + 1 < len(XG_) else None
        i15 = int(round(1.5/DX)); ret['value_at_1_5'] = [float(dF[0][i15]), float(sig[0][i15])]; i2 = int(round(2.0/DX)); ret['value_at_2'] = [float(dF[0][i2]), float(sig[0][i2])]
        ret['further_crossings'] = d0['further_crossings']
    # null models for the even characters: pure shift and pure scale of the model with the measured mean
    dm = R['plus']['x1_mean'] - R['plus']['x1_mean_model']; Fsh = np.interp(XG_ - dm, XG_, Fm[0], left=0.0); a_ = R['plus']['x1_mean']/R['plus']['x1_mean_model']; Fsc = np.interp(XG_/a_, XG_, Fm[0])
    nul = dict(shift=dict(delta=dm, max_dF=float(np.max(Fsh - Fm[0])), min_dF=float(np.min(Fsh - Fm[0])), crossings=roots(Fsh - Fm[0], 0.05, 2.9)),
               scale=dict(factor=a_, max_dF=float(np.max(Fsc - Fm[0])), min_dF=float(np.min(Fsc - Fm[0])), crossings=roots(Fsc - Fm[0], 0.05, 2.9)))
    R['even'] = dict(d0, crossing_se=se['crossing'], crossing_even_odd_se=se['crossing_even_odd'], second_crossing_even_odd_se=se['second_crossing_even_odd'], model_support_end=[0.98*R['plus']['N_eff'], 0.98*R['minus']['N_eff']], max_x_se=se['max_x'], min_x_se=se['min_x'], reenter_se=se['reenter'], **ret, null_models=nul,
                     odd_max_abs_dF=float(np.max(np.abs(dF[1]))), curve_step_0_02=[[float(XG_[i]), float(dF[0][i]), float(dF[1][i])] for i in range(0, len(XG_), 20)])
    if f in ('C100', 'B10'):
        Qp = J['families'][f]['plus']; pr = np.array(Qp['prediction_0025']); H = np.array(Qp['hist_0025'], float); ca = []
        for a in CAL_A:
            k_ = int(round(a/0.025)); dpr = float(pr[:k_].mean()); Ne_ = neff_a(dpr, a); Fm_ = model(Ne_); r_ = roots(Fd[0] - Fm_, 0.2, 1.2, -1)
            ca.append(dict(a=a, density_ratios=dpr, density_data=float(H[:k_].sum()/(nn[0]*a)), N_eff=Ne_, crossing=(r_[0] if r_ else None)))
        A_ = np.array([q['a'] for q in ca]); X_ = np.array([q['crossing'] for q in ca]); c1, c0 = np.polyfit(A_, X_, 1)
        R['calibration_test'] = dict(even=ca, slope=float(c1), intercept=float(c0), max_abs_crossing_minus_a=float(np.max(np.abs(X_ - A_))))
        print('%-5s calibration test: crossing %s for a = %s; slope %.3f, intercept %.3f' % (f, ' '.join('%.3f' % x for x in X_), ' '.join('%.1f' % a for a in A_), c1, c0), flush=True)
    OUT['families'][f] = R
    e = R['even']; print('%-5s even: crossing %.4f +- %.4f (even = odd %.4f +- %.4f); max %.3f at %.3f; min %.4f at %.3f +- %.3f; back in KS band %s +- %s, within 1 error from %s; dF(1.5) %.5f +- %.5f; further %s; even = odd again %s +- %s (model ends %.3f); shift null %s'
          % (f, e['crossing'] or -1, e['crossing_se'] or -1, e['crossing_even_odd'] or -1, e['crossing_even_odd_se'] or -1, e.get('max', 0), e.get('max_x', 0), e.get('min', 0), e.get('min_x', 0), e['min_x_se'] or -1,
             e.get('reenter_ks_band'), None if e['reenter_se'] is None else round(e['reenter_se'], 3), e.get('within_one_error_from'), e['value_at_1_5'][0], e['value_at_1_5'][1], e['further_crossings'][:3], e.get('second_crossing_even_odd'), None if e['second_crossing_even_odd_se'] is None else round(e['second_crossing_even_odd_se'], 3), e['model_support_end'][0], nul['shift']['crossings']),
          '| model check %.1e' % R['plus']['model_check_against_reference'], round(time.time() - t0), 's', flush=True)
json.dump(OUT, open(os.path.join(OUT_DIR, 'reference_first_zero_crossing.json'), 'w')); print('done', round(time.time() - t0), 's', flush=True)
