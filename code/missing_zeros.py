#!/usr/bin/env python3
"""
Low-lying zeros of quadratic Dirichlet L-functions, study III, Section 4.7: how many zeros the search of the data misses.

usage:  python3 -I missing_zeros.py ZEROS_ROOT FAMILY SAMPLE SEED OUT_JSON      (PARI/GP through cypari2; B10 3 000 L in about 4 minutes)

ks_nuly.py finds the zeros with lfunzeros(L, T), which looks for sign changes of the Hardy function with a step of the mean spacing divided by
divz = 8 (the default of PARI).  Two zeros closer than one step give no sign change and are both missed.  For a random sample of the family
(random.seed(SEED), random.sample) every L-function is recomputed with divz = 8 (it must equal the data) and with divz = 64; the zeros found only
with divz = 64 are the missed ones (their height t and unfolded position x = theta_D(t)/pi are recorded).
"""
import sys, os, math, json, time, random, numpy as np, cypari2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); import read_zeros as RZ
P = cypari2.Pari(); P.allocatemem(1024*10**6); P.set_real_precision(38)
def theta(D, T): k = 0 if D > 0 else 1; return 0.5*T*math.log(abs(D)/math.pi) + float(P.imag(P.lngamma(P('(1/2+%d)/2' % k) + P('I')*T/2)))
def tmax(D, x):
    T = 2.0
    for _ in range(80):
        f = theta(D, T)/math.pi - x
        if abs(f) < 1e-10: break
        h = 1e-5; d = (theta(D, T + h) - theta(D, T - h))/(2*h*math.pi); T -= f/d
    return T
ROOT, fam, nsamp, seed, OUT = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
ALL = [(D, g) for D, g in RZ.zeros(fam, ROOT)]; random.seed(seed); S = random.sample(range(len(ALL)), nsamp); t0 = time.time()
out = dict(family=fam, n_family=len(ALL), sample=nsamp, seed=seed, divz_data=8, divz_check=64, same_as_data=0, missing_L=0, missing_zeros=0, examples=[], missing_x=[])
for c, i in enumerate(S):
    D, g = ALL[i]; L = P.lfuncreate(int(D)); T = tmax(int(D), 4.35)
    z8 = [round(float(v), 10) for v in P.lfunzeros(L, T) if float(v) > 1e-9]; z64 = [float(v) for v in P.lfunzeros(L, T, 64) if float(v) > 1e-9]
    out['same_as_data'] += int(len(z8) == len(g) and all(abs(a - b) < 1e-6 for a, b in zip(z8, g)))
    if len(z64) != len(z8):
        out['missing_L'] += 1; out['missing_zeros'] += len(z64) - len(z8)
        miss = [v for v in z64 if all(abs(v - u) > 1e-6 for u in z8)]; xs = [theta(int(D), v)/math.pi for v in miss]; out['missing_x'] += xs
        if len(out['examples']) < 12: out['examples'].append(dict(D=int(D), t_missing=[round(v, 4) for v in miss], x_missing=[round(v, 4) for v in xs]))
    if c % 500 == 499: print(fam, c + 1, out['missing_L'], round(time.time() - t0), 's', flush=True)
out['seconds'] = round(time.time() - t0); json.dump(out, open(OUT, 'w'), indent=1)
print(fam, 'sample', nsamp, 'identical to data', out['same_as_data'], 'L with missed zeros', out['missing_L'], 'missed zeros', out['missing_zeros'], 'x of missed', np.round(sorted(out['missing_x']), 3).tolist(), round(time.time() - t0), 's', flush=True)
