#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAYAN ALFA V1008 · KATZ–SARNAK III — kontrola výstupu generátoru ks_nuly.py (nezávislá na PARI/GP; jen standardní knihovna Pythonu)
Použití:  python3 ks_kontrola.py [ks_vystup]
Kontroluje pro každou rodinu v manifestu: SHA-256 bloků, úplnost (každý základní diskriminant právě jednou — vlastním testem
bezčtvercovosti, ne přes PARI), že nuly jsou kladné a rostoucí, a že počet nul do x = 4 na jednu L-funkci odpovídá rozvinutí
(průměr kolem 4,0; Riemannova–von Mangoldtova formule pro L(s, χ_D) dává 4 v průměru, po jednotlivých L se liší o ±2).
Na konci vypíše souhrn, který stačí poslat (nebo nechat ve složce, laboratoř si ho přečte sama).
"""
import gzip, hashlib, json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ks_nuly import RODINY, nahodne_D, zakladni

def lngamma_im(a, t):
    """Im ln Γ(a + it) pro a > 0 Stirlingovou řadou s posunem (přesnost ~1e-12), bez PARI"""
    z = complex(a, t); s = 0j
    while z.real < 12: s -= complexlog(z); z += 1
    zi = 1/z; z2 = zi*zi
    st = (z - 0.5)*complexlog(z) - z + 0.5*math.log(2*math.pi) + zi*(1/12 - z2*(1/360 - z2*(1/1260 - z2*(1/1680 - z2/1188))))
    return (st + s).imag

def complexlog(z):
    return complex(math.log(abs(z)), math.atan2(z.imag, z.real))

def theta(D, T):
    k = 0 if D > 0 else 1
    return 0.5*T*math.log(abs(D)/math.pi) + lngamma_im((0.5 + k)/2, T/2)

def bezctv_usek(lo, hi):
    """příznaky bezčtvercovosti pro m = lo+1 … hi sítem přes p² (rychlé i pro |D| ~ 10^12)"""
    n = hi - lo; f = bytearray(b'\x01')*n; P = int(math.isqrt(hi)) + 1; s = bytearray(b'\x01')*(P + 1); s[0:2] = b'\x00\x00'
    for p in range(2, int(math.isqrt(P)) + 1):
        if s[p]: s[p*p::p] = bytearray(len(s[p*p::p]))
    for p in range(2, P + 1):
        if s[p]:
            q = p*p; st = ((lo + 1 + q - 1)//q)*q
            for m in range(st, hi + 1, q): f[m - lo - 1] = 0
    return f

def ocekavane_D(jm):
    F = RODINY[jm]
    if F['typ'] == 'nahodne': return set(nahodne_D(F))
    lo, hi = F['lo'], F['hi']; S = bezctv_usek(lo, hi); S4 = bezctv_usek(lo//4, hi//4 + 1); out = set()
    for a in range(lo + 1, hi + 1):
        for D in (a, -a):
            if D % 4 == 1 and S[a - lo - 1]: out.add(D)
            elif D % 4 == 0:
                q = D//4
                if q % 4 in (2, 3) and S4[abs(q) - lo//4 - 1]: out.add(D)
    return out

def main():
    V = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ks_vystup')
    M = json.load(open(os.path.join(V, 'manifest.json'), encoding='utf-8')); vse_ok = True; sh = {}
    print('Výstup:', V, '· formát', M.get('format'), '· xmax', M.get('xmax'), '· PARI', M.get('prostredi', {}).get('pari'))
    for jm, R in M['rodiny'].items():
        B = R['bloky']; sha_ok = 0; D_vse = []; chyb_nul = 0; pocty = []
        for bid, b in B.items():
            p = os.path.join(V, jm, b['soubor'])
            if not os.path.exists(p): continue
            if hashlib.sha256(open(p, 'rb').read()).hexdigest() != b['sha256']: continue
            sha_ok += 1
            for line in gzip.open(p, 'rt', encoding='utf-8'):
                o = json.loads(line); D = o['D']; g = o['g']; D_vse.append(D)
                if any(v <= 0 for v in g) or any(g[i] >= g[i + 1] for i in range(len(g) - 1)): chyb_nul += 1
                pocty.append(sum(1 for v in g if theta(D, v)/math.pi < 4))
        hotovo = len(B) == R.get('bloku_celkem') and sha_ok == len(B)
        dup = len(D_vse) - len(set(D_vse)); ocek = ocekavane_D(jm) if hotovo else None
        uplne = hotovo and set(D_vse) == ocek
        prum = sum(pocty)/len(pocty) if pocty else float('nan')
        ok = hotovo and uplne and dup == 0 and chyb_nul == 0 and 3.8 < prum < 4.2
        vse_ok &= ok if hotovo else True
        sh[jm] = dict(bloku=len(B), bloku_celkem=R.get('bloku_celkem'), sha_ok=sha_ok, L=len(D_vse), duplicit=dup, uplne=uplne, chybnych_rad=chyb_nul, nul_do_4_na_L=round(prum, 4), v_poradku=ok)
        print('  %-5s bloků %d/%s, SHA-256 %d, L %d, úplné %s, duplicit %d, chybných řad nul %d, nul do x = 4 na L %.3f → %s'
              % (jm, len(B), R.get('bloku_celkem'), sha_ok, len(D_vse), 'ano' if uplne else ('ne' if hotovo else 'rozpracováno'), dup, chyb_nul, prum, 'V POŘÁDKU' if ok else ('ROZPRACOVÁNO' if not hotovo else 'CHYBA')))
    json.dump(dict(format='MAYAN_V1008_KS_KONTROLA_v1', manifest_sha256=hashlib.sha256(open(os.path.join(V, 'manifest.json'), 'rb').read()).hexdigest(), rodiny=sh), open(os.path.join(V, 'kontrola.json'), 'w', encoding='utf-8'), ensure_ascii=False, indent=1)
    print('KONTROLA', 'V POŘÁDKU' if vse_ok else 'NAŠLA CHYBU', '· souhrn uložen do kontrola.json')

if __name__ == '__main__':
    main()
