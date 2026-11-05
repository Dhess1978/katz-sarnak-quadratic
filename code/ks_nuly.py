#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MAYAN ALFA V1008 · KATZ–SARNAK III — generátor nízkých nul kvadratických L-funkcí L(s, χ_D)
==========================================================================================
Počítá nuly L(s, χ_D) pro základní diskriminanty D (PARI/GP přes cypari2) do výšky, kde rozvinutí
x = θ_D(γ)/π dosáhne XMAX (výchozí 4,3), a ukládá je po blocích. Výstup čte laboratoř (verze 34.62 a další).

Krok hledání (34.68d): lfunzeros hledá změny znaménka Hardyho funkce s krokem = střední rozestup nul / divz.
S výchozím divz = 8 v PARI vynechá asi 0,03–0,1 % L-funkcí dvojici velmi blízkých nul (dvě změny znaménka
v jednom kroku; např. D = 66 552: nuly 2,3005 a 2,3218). Proto je výchozí --divz 32 (asi o 25 % pomalejší).
Data spočtená dřív mají divz 8; manifest si divz pamatuje a do složky s jiným divz skript nepíše.

Použití (v Terminálu, ve složce se skriptem):
  python3 ks_nuly.py --test                        # rychlá kontrola instalace (asi 10 s)
  python3 ks_nuly.py --odhad --jadra 8             # změří rychlost na tomto počítači a odhadne čas rodin
  python3 ks_nuly.py --rodiny dohoda --jadra 8     # dohodnutý rozsah: C10, D10, E, R
  python3 ks_nuly.py --rodiny velky --jadra 8      # velký rozsah: C100, D100, E10, R10
  python3 ks_nuly.py --rodiny male --jadra 8       # malé vodiče: B10 (|D| = 10^4 … 10^5, asi 10 minut)
  python3 ks_nuly.py --rodiny E100 --jadra 9       # pokračování E10 u 10^10 (asi 109 000 L, asi 14 h na 9 jádrech)
  python3 ks_nuly.py --rodiny E100 --divz 8 --jadra 9 --vystup SLOZKA   # dokončení běhu začatého starší verzí (divz 8)
  python3 ks_nuly.py --rodiny C100,E10 --jadra 6   # vybrané rodiny
  python3 ks_nuly.py --souhrn                      # stav výstupů a kontrola SHA-256

Přerušení (Ctrl+C, zavření víka, vypnutí) nevadí: hotové bloky zůstanou a znovu spuštěný skript je přeskočí.
Výstup: složka ks_vystup/ vedle skriptu (lze změnit --vystup), v ní <rodina>/blok_XXXXXX.jsonl.gz a manifest.json.
Do výstupu se nezapisuje nic o počítači ani o uživateli (jen verze Pythonu, PARI a typ systému).

Potřebuje: Python 3.8+ a cypari2 (viz README_CZ.md). Nic se nestahuje z internetu, nic se neodesílá.
"""
import argparse, gzip, hashlib, json, math, os, platform, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

VERZE_FORMATU = 'MAYAN_V1008_KS_NULY_v1'
XMAX_VYCH = 4.3
DIVZ_VYCH = 32          # krok lfunzeros = střední rozestup / divz (PARI má výchozí 8, s ním se vynechávají blízké dvojice nul)
DIVZ = DIVZ_VYCH        # nastaví main() a _init() v každém procesu
DIVZ_PRACE = [DIVZ_VYCH]
# rodiny: 'rozsah' = všechny základní diskriminanty s lo < |D| <= hi (obě znaménka); 'nahodne' = n náhodných z lo <= |D| < hi (pevné semínko)
RODINY = {
    'A':    dict(typ='rozsah', lo=1000, hi=3000, blok=2000, popis='kalibrace: táž rodina jako v laboratoři 34.60 a 34.61 (1 213 L)'),
    'B':    dict(typ='rozsah', lo=10000, hi=20000, blok=2000, popis='kalibrace: táž rodina jako v laboratoři 34.60 a 34.61 (6 074 L)'),
    'B10':  dict(typ='rozsah', lo=10000, hi=100000, blok=2000, popis='|D| = 10^4 ... 10^5 (asi 54 700 L; obsahuje B, test jevu rodiny B)'),
    'C10':  dict(typ='rozsah', lo=1000000, hi=1100000, blok=2000, popis='|D| = 10^6 ... 1,1·10^6 (asi 60 800 L)'),
    'D10':  dict(typ='rozsah', lo=100000000, hi=100010000, blok=200, popis='|D| = 10^8 ... 10^8 + 10^4 (asi 6 080 L)'),
    'E':    dict(typ='rozsah', lo=10000000000, hi=10000002000, blok=20, popis='|D| = 10^10 ... 10^10 + 2 000 (asi 1 220 L)'),
    'R':    dict(typ='nahodne', lo=1000000, hi=10000000, n=6000, seed=3462, blok=250, popis='6 000 náhodných základních diskriminantů s |D| = 10^6 ... 10^7 (test χ² pod 0,95)'),
    'C100': dict(typ='rozsah', lo=1000000, hi=2000000, blok=2000, popis='|D| = 10^6 ... 2·10^6 (asi 608 000 L)'),
    'D100': dict(typ='rozsah', lo=100000000, hi=100100000, blok=200, popis='|D| = 10^8 ... 10^8 + 10^5 (asi 60 800 L)'),
    'E10':  dict(typ='rozsah', lo=10000000000, hi=10000020000, blok=20, popis='|D| = 10^10 ... 10^10 + 2·10^4 (asi 12 200 L)'),
    'E100': dict(typ='rozsah', lo=10000020000, hi=10000200000, blok=200, popis='|D| = 10^10 + 2·10^4 ... 10^10 + 2·10^5 (asi 109 000 L; navazuje na E10, spolu 10^10 ... 10^10 + 2·10^5)'),
    'R10':  dict(typ='nahodne', lo=1000000, hi=10000000, n=60000, seed=34620, blok=250, popis='60 000 náhodných základních diskriminantů s |D| = 10^6 ... 10^7'),
    'F':    dict(typ='rozsah', lo=100000000000, hi=100000000500, blok=4, popis='|D| = 10^11 ... 10^11 + 500 (asi 300 L; pomalé, volitelné)'),
    'G':    dict(typ='rozsah', lo=1000000000000, hi=1000000000200, blok=2, popis='|D| = 10^12 ... 10^12 + 200 (asi 120 L; velmi pomalé, volitelné)'),
}
SADY = {'kalibrace': ['A', 'B'], 'dohoda': ['C10', 'D10', 'E', 'R'], 'male': ['B10'], 'velky': ['C100', 'D100', 'E10', 'R10'], 'e100': ['E100'], 'vse': ['B10', 'C100', 'D100', 'E10', 'R10', 'F', 'G']}
# referenční nuly (PARI/GP 2.17, laboratoř 34.61) pro --test: první dvě nuly γ
TEST_NULY = {1001: [1.1982745987, 2.0940868414], -1003: [0.4187362863, 1.3160048641], 10001: [0.7491719725, 1.5410881529],
             -10003: [0.373690799, 0.8067250808], 1000001: [0.2339733287, 1.1183600986], -1000003: [0.1087097434, 0.6242220463]}
# blízká dvojice nul (34.68d): s divz 8 ji lfunzeros vynechá, s divz >= 16 ji najde
TEST_PAR = (66552, [0.3264, 0.8842, 2.3005, 2.3218])

_P = None   # instance PARI v každém pracovním procesu

def _init(divz=None):
    global _P, DIVZ
    if divz is not None: DIVZ = divz
    import cypari2
    _P = cypari2.Pari()
    try: _P.allocatemem(64*2**20, 2*2**30, silent=True)      # zásobník roste jen podle potřeby (u |D| ~ 10^10 asi 256 MB)
    except TypeError: _P.allocatemem(512*2**20)
    try: _P.default('nbthreads', 1)
    except Exception: pass

def theta(D, T):
    """θ_D(T) = T/2 · ln(|D|/π) + Im lnΓ((½ + κ)/2 + iT/2), κ = 0 pro D > 0, 1 pro D < 0"""
    k = 0 if D > 0 else 1
    return 0.5*T*math.log(abs(D)/math.pi) + float(_P.imag(_P.lngamma(_P('(1/2+%d)/2' % k) + _P('I')*T/2)))

def tmax(D, x):
    T = 2.0
    for _ in range(80):
        f = theta(D, T)/math.pi - x
        if abs(f) < 1e-10: break
        h = 1e-5; d = (theta(D, T + h) - theta(D, T - h))/(2*h*math.pi); T -= f/d
    return T

def nuly(D, xmax):
    return [round(float(v), 10) for v in _P.lfunzeros(_P.lfuncreate(D), tmax(D, xmax + 0.05), DIVZ) if float(v) > 1e-9]

def splitmix(seed):
    s = seed & 0xFFFFFFFFFFFFFFFF
    while True:
        s = (s + 0x9E3779B97F4A7C15) & 0xFFFFFFFFFFFFFFFF; z = s
        z = ((z ^ (z >> 30))*0xBF58476D1CE4E5B9) & 0xFFFFFFFFFFFFFFFF; z = ((z ^ (z >> 27))*0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF
        yield z ^ (z >> 31)

def bezctv(m):
    m = abs(m); p = 2
    while p*p <= m:
        if m % (p*p) == 0: return False
        p += 1 if p == 2 else 2
    return True

def zakladni(D):
    if D % 4 == 1: return bezctv(D)
    if D % 4 == 0: q = D//4; return q % 4 in (2, 3) and bezctv(q)
    return False

def nahodne_D(F):
    g = splitmix(F['seed']); out = set(); span = F['hi'] - F['lo']
    while len(out) < F['n']:
        r = next(g); a = F['lo'] + r % span; D = a if (r >> 63) & 1 else -a
        if zakladni(D): out.add(D)
    return sorted(out, key=lambda d: (abs(d), d))

def bloky(jm):
    """seznam (id_bloku, popis) — u rozsahů úsek |D|, u náhodných úsek seznamu"""
    F = RODINY[jm]
    if F['typ'] == 'rozsah':
        return [(a, (a, min(a + F['blok'], F['hi']))) for a in range(F['lo'], F['hi'], F['blok'])]
    L = nahodne_D(F); return [(i, L[i:i + F['blok']]) for i in range(0, len(L), F['blok'])]

def pocitej_blok(jm, bid, spec, xmax, cesta):
    if _P is None: _init()
    assert DIVZ == DIVZ_PRACE[0], 'divz procesu nesouhlasí'
    t0 = time.time(); rad = []
    if isinstance(spec, tuple):
        Ds = [D for a in range(spec[0] + 1, spec[1] + 1) for D in (a, -a) if _P.isfundamental(D)]
    else:
        Ds = [D for D in spec if _P.isfundamental(D)]
    nn = 0
    for D in Ds:
        g = nuly(D, xmax); nn += len(g); rad.append(json.dumps({'D': D, 'g': g}, separators=(',', ':')))
    data = ('\n'.join(rad) + '\n').encode('utf-8'); tmp = cesta + '.tmp'
    with gzip.GzipFile(tmp, 'wb', mtime=0) as f: f.write(data)
    os.replace(tmp, cesta)
    return dict(jm=jm, bid=bid, soubor=os.path.basename(cesta), sha256=hashlib.sha256(open(cesta, 'rb').read()).hexdigest(), nL=len(Ds), nnul=nn, sekund=round(time.time() - t0, 2))

def _init_prace(divz):
    DIVZ_PRACE[0] = divz; _init(divz)

def manifest_cti(V):
    p = os.path.join(V, 'manifest.json')
    return json.load(open(p, encoding='utf-8')) if os.path.exists(p) else None

def manifest_pis(V, M):
    p = os.path.join(V, 'manifest.json'); tmp = p + '.tmp'
    json.dump(M, open(tmp, 'w', encoding='utf-8'), ensure_ascii=False, indent=1); os.replace(tmp, p)

def prostredi():
    import cypari2
    _init()
    return dict(python=platform.python_version(), system=platform.system(), stroj=platform.machine(), pari=str(_P.version()), cypari2=getattr(cypari2, '__version__', '?'),
                skript_sha256=hashlib.sha256(open(os.path.abspath(__file__), 'rb').read()).hexdigest())

def test():
    _init(); chyba = 0
    for D, ref in TEST_NULY.items():
        g = nuly(D, 4.3)[:2]; d = max(abs(a - b) for a, b in zip(g, ref)) if len(g) == 2 else 1
        chyba = max(chyba, d); print('  D = %d: %s (reference %s) %s' % (D, g, ref, 'OK' if d < 1e-8 else 'ROZDÍL'))
    D, ref = TEST_PAR; g = [v for v in nuly(D, 4.3) if v < 3.1]; ok = len(g) == len(ref) and max(abs(a - b) for a, b in zip(g, ref)) < 1e-4
    print('  D = %d (blízká dvojice nul, divz %d): %s %s' % (D, DIVZ, [round(v, 4) for v in g], 'OK' if ok else ('vynechaná dvojice — s divz 8 se to čeká' if DIVZ < 16 else 'ROZDÍL')))
    if DIVZ >= 16 and not ok: chyba = 1
    print('TEST', 'v pořádku' if chyba < 1e-8 else 'NESOUHLASÍ — pošlete prosím tento výpis', '(největší rozdíl %.1e)' % chyba)
    return chyba < 1e-8

def odhad(jadra):
    _init(); print('Odhad rychlosti (měří se na několika L z každé rodiny; skutečný čas se může lišit o desítky procent):')
    for jm in ['C10', 'D10', 'E', 'R', 'B10', 'C100', 'D100', 'E10', 'E100', 'R10', 'F', 'G']:
        F = RODINY[jm]; k = {'F': 1, 'G': 1, 'E': 3, 'E10': 3, 'E100': 3}.get(jm, 8 if F['lo'] < 10**9 else 4); t0 = time.time(); n = 0
        a = F['lo'] + 1 if F['typ'] == 'rozsah' else (F['lo'] + F['hi'])//2
        while n < k:
            for D in (a, -a):
                if n < k and _P.isfundamental(D): nuly(D, XMAX_VYCH); n += 1
            a += 1
        sL = (time.time() - t0)/n
        nL = round(F['n'] if F['typ'] == 'nahodne' else 6/math.pi**2*(F['hi'] - F['lo']))   # 3/π² na znaménko
        h = sL*nL/3600/max(jadra, 1); print('  %-5s %-70s %8.2f s na L  · asi %7d L · %6.1f h na %d jádrech' % (jm, F['popis'], sL, nL, h, jadra))
        if jm in ('F', 'G') and h > 12: print('        (rodina %s je na tomto počítači velmi pomalá — doporučuji vynechat)' % jm)

def souhrn(V):
    M = manifest_cti(V)
    if not M: print('Ve složce', V, 'zatím není manifest.json.'); return
    for jm, R in M['rodiny'].items():
        B = R['bloky']; ok = 0; nL = 0; nn = 0
        for bid, b in B.items():
            p = os.path.join(V, jm, b['soubor'])
            if os.path.exists(p) and hashlib.sha256(open(p, 'rb').read()).hexdigest() == b['sha256']: ok += 1; nL += b['nL']; nn += b['nnul']
        print('  %-5s bloků %d / %d (SHA-256 v pořádku %d), L %d, nul %d %s' % (jm, len(B), R['bloku_celkem'], ok, nL, nn, '— HOTOVO' if ok == R['bloku_celkem'] else ''))

def main():
    ap = argparse.ArgumentParser(description='MAYAN ALFA V1008 · KATZ–SARNAK III — nuly kvadratických L-funkcí (PARI/GP)')
    ap.add_argument('--rodiny', default='', help='sada (kalibrace, dohoda, male, velky, vse) nebo seznam rodin oddělený čárkou: ' + ', '.join(RODINY))
    ap.add_argument('--jadra', type=int, default=max(1, (os.cpu_count() or 2) - 1), help='počet souběžných procesů (výchozí: počet jader − 1)')
    ap.add_argument('--vystup', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ks_vystup'), help='výstupní složka')
    ap.add_argument('--xmax', type=float, default=XMAX_VYCH, help='rozvinutá výška x = θ_D(γ)/π, do které se počítá (výchozí 4,3)')
    ap.add_argument('--divz', type=int, default=DIVZ_VYCH, help='krok hledání lfunzeros = střední rozestup / divz (výchozí 32; data před 34.68d mají 8)')
    ap.add_argument('--test', action='store_true'); ap.add_argument('--odhad', action='store_true'); ap.add_argument('--souhrn', action='store_true')
    a = ap.parse_args(); V = a.vystup
    global DIVZ; DIVZ = a.divz; DIVZ_PRACE[0] = a.divz
    if a.test: sys.exit(0 if test() else 1)
    if a.odhad: odhad(a.jadra); return
    if a.souhrn: souhrn(V); return
    if not a.rodiny: ap.print_help(); return
    jm_list = SADY.get(a.rodiny) or [x.strip() for x in a.rodiny.split(',') if x.strip()]
    for jm in jm_list:
        if jm not in RODINY: sys.exit('Neznámá rodina %s (možnosti: %s)' % (jm, ', '.join(RODINY)))
    if not test(): sys.exit('Kontrola instalace nesouhlasí — výpočet se nespouští.')
    os.makedirs(V, exist_ok=True)
    M = manifest_cti(V) or dict(format=VERZE_FORMATU, prostredi=prostredi(), xmax=a.xmax, divz=a.divz, rodiny={})
    if abs(M.get('xmax', a.xmax) - a.xmax) > 1e-12: sys.exit('Výstupní složka už obsahuje data s jiným --xmax (%s); zvolte jinou --vystup.' % M['xmax'])
    if int(M.get('divz', 8)) != a.divz: sys.exit('Výstupní složka už obsahuje data s divz %d (bez záznamu = 8, verze před 34.68d); pro dokončení zadejte --divz %d, pro nový výpočet jinou --vystup.' % (int(M.get('divz', 8)), int(M.get('divz', 8))))
    M['divz'] = a.divz
    M['prostredi_posledni'] = prostredi()
    ulohy = []
    for jm in jm_list:
        os.makedirs(os.path.join(V, jm), exist_ok=True); B = bloky(jm); R = M['rodiny'].setdefault(jm, dict(spec={k: v for k, v in RODINY[jm].items()}, bloky={}))
        R['bloku_celkem'] = len(B)
        for bid, spec in B:
            cesta = os.path.join(V, jm, 'blok_%06d.jsonl.gz' % (bid if RODINY[jm]['typ'] == 'nahodne' else bid//RODINY[jm]['blok']))
            hot = R['bloky'].get(str(bid))
            if hot and os.path.exists(cesta) and hashlib.sha256(open(cesta, 'rb').read()).hexdigest() == hot['sha256']: continue
            ulohy.append((jm, bid, spec, cesta))
    manifest_pis(V, M)
    print('Úloh (bloků) ke spočtení: %d, procesů: %d, výstup: %s' % (len(ulohy), a.jadra, V), flush=True)
    if not ulohy: souhrn(V); return
    t0 = time.time(); hotovo = 0; nL = 0
    ctx = mp.get_context('spawn')
    try:
        with ProcessPoolExecutor(max_workers=a.jadra, mp_context=ctx, initializer=_init_prace, initargs=(a.divz,)) as ex:
            fut = [ex.submit(pocitej_blok, jm, bid, spec, a.xmax, cesta) for jm, bid, spec, cesta in ulohy]
            for f in as_completed(fut):
                r = f.result(); hotovo += 1; nL += r['nL']
                M['rodiny'][r['jm']]['bloky'][str(r['bid'])] = {k: r[k] for k in ('soubor', 'sha256', 'nL', 'nnul', 'sekund')}
                if hotovo % 5 == 0 or hotovo == len(ulohy): manifest_pis(V, M)
                el = time.time() - t0; zb = el/hotovo*(len(ulohy) - hotovo)
                print('  %s blok %s: %d L, %d nul, %.0f s · celkem %d/%d bloků, %d L, zbývá asi %.1f h' % (r['jm'], r['bid'], r['nL'], r['nnul'], r['sekund'], hotovo, len(ulohy), nL, zb/3600), flush=True)
    except KeyboardInterrupt:
        print('\nPřerušeno — hotové bloky jsou uložené, příště se přeskočí.'); manifest_pis(V, M); sys.exit(1)
    manifest_pis(V, M); print('Hotovo za %.1f h.' % ((time.time() - t0)/3600)); souhrn(V)

if __name__ == '__main__':
    main()
