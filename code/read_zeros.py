#!/usr/bin/env python3
"""
Reader for the zeros of quadratic Dirichlet L-functions L(s, chi_D) produced by ks_nuly.py (format MAYAN_V1008_KS_NULY_v1).
Standard library only; nothing is modified.

Layout of a store (e.g. the folder written by ks_nuly.py, or several such folders side by side):
    ROOT/<set>/manifest.json                    format, xmax, environment, families -> spec, blocks {id: file, sha256, nL, nnul}
    ROOT/<set>/<family>/blok_NNNNNN.jsonl.gz     one line per L-function: {"D": fundamental discriminant, "g": [gamma_1, gamma_2, ...]}
The heights gamma are the positive ordinates of the zeros on the critical line, increasing, stored to 10 decimals, up to the unfolded
height x = theta_D(gamma)/pi = xmax (4.3), i.e. about four zeros per L-function.

    from read_zeros import families, zeros, unfold, theta
    for D, g in zeros('C100', root):              # D and the heights gamma (SHA-256 of every block is checked against the manifest)
    for D, x in zeros('E10', root, unfolded=True) # unfolded zeros x = theta_D(gamma)/pi (mean density 1)
    for D, g in study_zeros('E100', root)         # a family of the study: E100 = the generator families E10 and E100 (Section 4.8)
"""
import gzip, hashlib, json, math, os, sys

FORMAT = 'MAYAN_V1008_KS_NULY_v1'

def _clog(z):
    return complex(math.log(abs(z)), math.atan2(z.imag, z.real))

def _lngamma_im(a, t):
    """Im ln Gamma(a + it), Stirling series after shifting the argument (accuracy ~1e-12)"""
    z = complex(a, t); s = 0j
    while z.real < 12: s -= _clog(z); z += 1
    zi = 1/z; z2 = zi*zi
    st = (z - 0.5)*_clog(z) - z + 0.5*math.log(2*math.pi) + zi*(1/12 - z2*(1/360 - z2*(1/1260 - z2*(1/1680 - z2/1188))))
    return (st + s).imag

def theta(D, t):
    """theta_D(t) = (t/2) ln(|D|/pi) + Im ln Gamma((1/2 + k)/2 + it/2), k = 0 for D > 0 (even character), k = 1 for D < 0 (odd)"""
    k = 0 if D > 0 else 1
    return 0.5*t*math.log(abs(D)/math.pi) + _lngamma_im((0.5 + k)/2, t/2)

def unfold(D, g):
    return [theta(D, t)/math.pi for t in g]

def sets(root):
    """{set: manifest} for every subfolder of root with a manifest.json of the format MAYAN_V1008_KS_NULY_v1 (alphabetical)"""
    out = {}
    for d in sorted(os.listdir(root)):
        p = os.path.join(root, d, 'manifest.json')
        if not os.path.isfile(p): continue
        try: M = json.load(open(p, encoding='utf-8'))
        except ValueError: continue
        if M.get('format') == FORMAT: out[d] = M
    return out

def families(root):
    """{family: [set, ...]}"""
    out = {}
    for s, M in sets(root).items():
        for f in M['rodiny']: out.setdefault(f, []).append(s)
    return out

def zeros(family, root, unfolded=False, check=True, set_=None):
    """generator of (D, zeros) in the order of the blocks"""
    S = sets(root); F = families(root)
    if family not in F: raise KeyError('family %s not found in %s (available: %s)' % (family, root, ', '.join(sorted(F))))
    s = set_ or F[family][0]; R = S[s]['rodiny'][family]
    if R.get('bloku_celkem') is not None and len(R['bloky']) != R['bloku_celkem']:
        print('warning: family %s in %s is incomplete (%d of %d blocks)' % (family, s, len(R['bloky']), R['bloku_celkem']), file=sys.stderr)
    for bid, b in sorted(R['bloky'].items(), key=lambda kv: int(kv[0])):
        data = open(os.path.join(root, s, family, b['soubor']), 'rb').read()
        if check and hashlib.sha256(data).hexdigest() != b['sha256']: raise ValueError('SHA-256 of block %s/%s does not match the manifest' % (family, b['soubor']))
        for line in gzip.decompress(data).decode('utf-8').splitlines():
            if line.strip():
                o = json.loads(line); D, g = o['D'], o['g']
                yield D, (unfold(D, g) if unfolded else g)

# families of the study made of several generator families (Section 4.8): E100 = |D| = 10^10 ... 10^10 + 2*10^5 = E10 (to 10^10 + 2*10^4) + E100 (the rest)
STUDY = {'E100': ('E10', 'E100')}

def study_zeros(family, root, unfolded=False, check=True):
    """generator of (D, zeros) of a family of the study, in the order of the generator families and their blocks"""
    for part in STUDY.get(family, (family,)):
        yield from zeros(part, root, unfolded=unfolded, check=check)

def content_sha256(root):
    """SHA-256 of the decompressed content of every block (the compressed files differ between zlib versions, the content does not)"""
    out = []
    for s, M in sets(root).items():
        for f, R in M['rodiny'].items():
            for bid, b in sorted(R['bloky'].items(), key=lambda kv: int(kv[0])):
                raw = gzip.decompress(open(os.path.join(root, s, f, b['soubor']), 'rb').read())
                out.append((f, int(bid), b['nL'], b['nnul'], hashlib.sha256(raw).hexdigest()))
    return out

if __name__ == '__main__':
    root = sys.argv[1] if len(sys.argv) > 1 else '.'
    for s, M in sets(root).items():
        print(s, '· xmax', M.get('xmax'), '· PARI', M.get('prostredi', {}).get('pari'))
        for f, R in M['rodiny'].items():
            B = R['bloky']; print('   %-5s %s · blocks %d/%s · L %d · zeros %d' % (f, R['spec'].get('popis', ''), len(B), R.get('bloku_celkem'), sum(b['nL'] for b in B.values()), sum(b['nnul'] for b in B.values())))
