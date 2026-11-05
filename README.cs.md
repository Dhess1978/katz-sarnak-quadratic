# Nízké nuly 795 381 kvadratických Dirichletových L-funkcí do vodiče 10¹⁰: domněnka poměrů, souvislé rodiny a první nula (studie III)

**David Hess (MAYAN ALFA V1008 LAB v.34.68e) · 5.11.2026** · [English version](README.md)

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23991165.svg)](https://doi.org/10.5281/zenodo.23991165) Zenodo: [doi:10.5281/zenodo.23991165](https://doi.org/10.5281/zenodo.23991165)

Třetí studie řady, po studii *Berryho saturace a statistika nul Riemannovy zeta funkce* (Zenodo, [doi:10.5281/zenodo.23815328](https://doi.org/10.5281/zenodo.23815328)) a *Za diagonální aproximací* (studie II; Zenodo, [doi:10.5281/zenodo.23928017](https://doi.org/10.5281/zenodo.23928017)). Přechází od Riemannovy zeta funkce k rodině kvadratických Dirichletových L-funkcí L(s, χ_D) z oddílu 6 přehledu Berryho a Keatinga (1999): jedenáct rodin základních diskriminantů od |D| ≈ 10³ do 10¹⁰, 795 381 L-funkcí, nízké nuly spočtené v PARI/GP na dvou počítačích (69 452 L-funkcí spočtených na obou má shodné nuly). Jednohladinová hustota rozvinutých nul x = θ_D(γ)/π se srovnává s limitou Katze a Sarnaka USp, s konečnou maticí USp(2N) a s domněnkou poměrů Conreyho, Farmera a Zirnbauera ve tvaru Conreyho a Snaithové (2007), vyčíslenou pro každý diskriminant bez volného parametru.

| výsledek | hodnota |
|---|---|
| jednohladinová hustota, χ² na koš (koše 0,05, x ≤ 4, poissonovská chyba pozorovaného počtu), domněnka poměrů, 11 rodin / očekávání pro nezávislé L-funkce | 0,54 – 1,19 / asi 0,95 |
| totéž proti limitě USp (rodiny s nejméně 6 000 L-funkcemi) | 5 – 850 |
| hustota v [0; ½], 607 941 L-funkcí s 10⁶ < \|D\| ≤ 2·10⁶: změřeno / domněnka poměrů / USp | 0,6082 ± 0,0014 / 0,6084 / 0,411 |
| totéž při \|D\| ≈ 10¹⁰ (12 153 L-funkcí) | 0,527 ± 0,009 / 0,527 / 0,411 (USp 12,6 chyby daleko) |
| výkon na nízkých frekvencích (kosinové módy k = 1 … 7, u < 1) pásem 6 074 po sobě jdoucích diskriminantů vůči náhodným výběrům stejné velikosti, \|D\| ≈ 5·10⁴ / 1,5·10⁶ / 10⁸ | 0,23 / 0,36 / 0,48 (5,9 / 10,2 / 4,3 chyby) |
| totéž na vysokých frekvencích (k = 32 … 79) | bez rozdílu |
| stratifikace náhodných výběrů podle D mod 840: snížení výkonu na nízkých frekvencích | 14 – 16 % v prvních výběrech (pětina až čtvrtina deficitu); se čtyřikrát až pětkrát více výběry 18 % v B10, v C100 a D100 nevýznamné |
| stratifikace podle (znaménko, D mod M) pro 19 modulů (3 … 840 a 131) a podle samotného znaménka, s nezávislým opakováním | jev nevytvoří: C100 a D100 jako náhodné výběry, B10 jen mod 840 (z = −5,1) |
| přímka náhodnými výběry a souvislými pásmy: přesný zbylý rozptyl W₁₀₀ součtů χ_D(p), p ≤ 100 (bez nul) → výkon nul na nízkých frekvencích všech 60 stratifikovaných podmnožin | směr souhlasí; velikost na hraně: χ² = 82,05 pro 60 bodů se společnými chybami (p = 0,03), 42 z 60 odchylek kladných |
| profil r_p souvislých pásem (Δ ≈ 10⁴) ve třech rodinách, 582 prvočísel do 4·10⁶ | táž funkce p: 0,04–0,05 pro p ≤ 100, 0,84–0,90 do 10⁵, nad tím asi 1 |
| první nula, USp(2N) s N = ½ ln(\|D\|/π): statistika KS / kritická hodnota | 1,6 – 53 |
| první nula, USp(2N_ef) s N_ef z domněnky poměrů, liché charaktery | KS pod kritickou hodnotou u všech vodičů |
| totéž, sudé charaktery: první nula výš o | asi 0,094/ln(\|D\|/π) (14,8 chyby v rodině 607 941) |
| totéž, sudé charaktery: průsečík rozdělení s USp(2N_ef), C100 | 0,508 ± 0,021; sleduje konec a intervalu [0; a], na kterém se N_ef určuje (sklon 1,13 pro a = 0,3 … 0,7), tedy to není vlastnost nul |
| přesné očekávání výkonu návrhu podmnožin na nízkých frekvencích bez losování (vrstva 0); statistika testu nad ním u náhodných výběrů, B10 / C100 / D100 | 2,7 / 3,5 / 1,9 %, kladně u všech 30 návrhů: zkreslení dvou nejnižších košů (v B10 z 84 %), ne šum |
| trojice po sobě jdoucích diskriminantů / náhodné výběry | 0,972 v B10 (4,7 chyby), 0,975 v C100 (4,3) |
| korelace kosinových módů 1 a 7 jedné L-funkce, B10 / C100 / D100 (poissonovské nuly: 0) | 0,58 / 0,47 / 0,39 |
| přírůstek vysvětleného podílu módu 7 díky χ_D(n) 16 složených n < e⁴, B10 / C100 / D100 (tytéž veličiny přeházené) | 0,188 / 0,111 / 0,043 (nejvýš 0,0003) |
| přepočet každého losovaného čísla oddílů 4.3–4.5 po hladinách (oddíl 4.7), stejné návrhy: chyba výkonu nízkých frekvencí / kontrola vrstvy 0 | 1,6 – 4,6× menší (medián 3,7) / střední z² 1,13 pro 84 návrhů; původní hodnoty souhlasí v rámci společných chyb |
| totéž: stratifikace podle D mod 840 proti náhodným výběrům, B10 / C100 / D100 | z = −26,6 / −13,5 / −2,9; 15 / 16 / 3 z 19 modulů za 3 chybami, ale malé (17 / 9,5 / 1,6 % náhodné hodnoty) |
| přímka náhodnými výběry a pásmy s hodnotami po hladinách | χ² = 841,8 pro 60: ve velikosti zamítnuto; přesně 0,89 / 0,56 / 0,39 snížení přímky |
| váhy χ_D(p) v příspěvku jedné L-funkce, sklon fit / explicitní formule: diagonální členy / s duálním členem domněnky poměrů | 0,52 – 0,60 / 0,995 – 0,999 |
| průměr první nuly sudých charakterů minus USp(2N_ef(a)), N_ef určené na [0; a], C100, a = 0,3 / 0,5 / 0,7 | +0,0153 / +0,0072 / −0,0012 (± 0,0005): závisí na kalibraci; tvar (KS) se v B10 a C100 odchyluje při každém a |
| nuly vynechané hledáním (krok / 8) ve 3 000 L-funkcích, B10 / C100 | jedna dvojice v jedné L-funkci / žádná |
| rodina E100 u \|D\| = 10¹⁰ (oddíl 4.8), 121 557 L-funkcí: χ² na koš proti domněnce poměrů / bez A a Δ / USp | 0,759 / 16,5 / 66,7; hustota v [0, ½] 0,5291 ± 0,0030 (domněnka poměrů 0,5267, USp 0,4105) |
| jeden souvislý úsek / náhodné výběry, přesně, B10 / C100 / D100 / E100 | 0,204 / 0,391 / 0,606 / 0,622; podíl snížení předpovězený jednotlivými prvočísly 89 / 76 / 65 / 42 % |
| E100, průměr první nuly minus USp(2N_ef), sudé / liché; tvar F(1) − USp(2N_ef) sudých charakterů | +0,0016 / −0,0009 (± 0,0012), 2,3 chyby pod 0,094/l; −0,0029 ± 0,0015 proti −0,0069 z B10–D100: klesá rychleji než 1/l |

Domněnka poměrů i limita Katze a Sarnaka jsou známá teorie; přínosem je soustavný, reprodukovatelný test na velkém souboru nul s nulovými modely a nezávislou kontrolou, rozpoznání nízkého χ² souvislých rodin jako vlastnosti výběru rodiny (úplné periody χ_D(p) pro prvočísla menší než délka úseku, viditelné přes explicitní vzorec na nízkých frekvencích) a jeho předpověď ze součtů charakterů bez nul rozdíl sudých a lichých charakterů u první nuly, přesné očekávání návrhů podmnožin bez losování a vazba kosinových módů přes součiny charakterů malých prvočísel. Oddíl 6 článku odděluje, co je, pokud jsme mohli zjistit, nové a co potvrzené. Nic z toho se netýká zobecněné Riemannovy hypotézy.

## Obsah

| složka | co |
|---|---|
| `paper/` | článek: `katz_sarnak_quadratic_en.pdf`, zdroj LaTeX `main.tex`, verze Markdown a DOCX; česká verze v `paper/cs/` |
| `figures/` | 10 obrázků, anglicky (`_en`) a česky (`_cz`), PNG 300 dpi a PDF (vektor) |
| `data/` | `reference_families.json`, `reference_subsets.json`, `reference_pieces.json`, `reference_pieces_repeat.json`, `reference_moduli.json`, `reference_characters.json`, `reference_layers.json`, `reference_intermodulation.json`, `reference_first_zero_crossing.json`, `reference_recomputation.json`, `reference_explicit_weights.json`, `reference_two_level.json`, `reference_missing_zeros_B10.json`, `reference_missing_zeros_C100.json` (všechna čísla oddílu 4), exporty CSV po rodinách, koších, podmnožinách, kusech (s chybou každého bodu), modulech, součtech charakterů, prvočíslech profilu pásem, návrzích vrstvy 0, módech intermodulace a průsečíku první nuly, přepočtu po hladinách, vahách explicitní formule, dvouhladinové části první nuly a vynechaných nulách, `reference_families_E100.json`, `reference_recomputation_E100.json`, `reference_two_level_E100.json`, `reference_explicit_weights_E100.json` a `reference_e100_trends.json` (oddíl 4.8, rodina E100; E100 je i v exportech CSV, trendy podle vodiče v `e100_trends.csv`), `block_content_sha256.csv` (SHA-256 rozbaleného obsahu každého bloku nul), manifesty pěti sad nul |
| `code/` | `ks_nuly.py` (generátor nul, PARI/GP přes cypari2), `ks_kontrola.py` (kontrola úplnosti bez PARI), `read_zeros.py`, `compute_families.py`, `pieces_repeat.py`, `compute_moduli.py`, `character_sums.py`, `exact_layers.py`, `intermodulation.py`, `first_zero_crossing.py`, `layered_recomputation.py`, `explicit_weights.py`, `two_level_first_zero.py`, `missing_zeros.py`, `independent_check.py`, `figures.py`, `figures_moduli.py`, `figures_layers.py`, `figures_recomputation.py`, `e100_trends.py`, `figures_e100.py`, `export_csv.py` |

## Reprodukce

Nuly přiloženy nejsou (3 806 316 nul v jedenácti rodinách, 3 452 121 různých, a 471 278 v 900 nových blocích E100 z oddílu 4.8; asi 35 MB komprimovaně); znovu je vytvoří `code/ks_nuly.py`, skript, který je spočetl (`--help` vypíše volby). Manifesty zaznamenávají SHA-256 revize generátoru, která danou sadu spočetla; revize do verze 34.68c se liší jen seznamem rodin a odhadem doby běhu; současná revize přidává volbu `--divz` (krok hledání = střední rozestup / divz, výchozí 32, oddíl 4.7 článku) a s `--divz 8` vytvoří nuly nejstarší sady (rodiny A a B) totožně.

```
pip install -r requirements.txt          # cypari2 (PARI/GP) jen pro ks_nuly.py, missing_zeros.py a kontroly 3 a 16 v independent_check.py
python3 code/ks_nuly.py --test
python3 code/ks_nuly.py --rodiny A,B,B10,C10,C100,D10,D100,E,E10,R,R10 --jadra 8 --divz 8 --vystup ZEROS_ROOT/set1   # zveřejněná data: krok = střední rozestup / 8
python3 code/ks_kontrola.py ZEROS_ROOT/set1                                                 # úplnost a formát, bez PARI
python3 -I code/compute_families.py ZEROS_ROOT data/                                        # asi 5 minut
python3 -I code/pieces_repeat.py ZEROS_ROOT data/                                           # test kusy, 400 podmnožin na bod s chybami, asi 4 minuty
python3 -I code/compute_moduli.py ZEROS_ROOT data/reference_moduli.json                     # oddíl 4.4, asi 8 minut
python3 -I code/character_sums.py ZEROS_ROOT data/reference_moduli.json data/reference_characters.json   # asi 20 minut
python3 -I code/exact_layers.py ZEROS_ROOT data/                                           # oddíl 4.6, asi 40 minut, 4 GB RAM
python3 -I code/intermodulation.py ZEROS_ROOT data/                                        # intermodulace a filtry, asi 2 minuty
python3 -I code/first_zero_crossing.py ZEROS_ROOT data/reference_families.json data/       # průsečík první nuly s testem kalibrace
python3 -I code/layered_recomputation.py ZEROS_ROOT data/ data/                           # oddíl 4.7 po hladinách, asi 25 minut, 4 GB RAM
python3 -I code/explicit_weights.py ZEROS_ROOT data/                                       # explicitní formule s duálním členem, asi 20 minut, 4 GB RAM
python3 -I code/two_level_first_zero.py ZEROS_ROOT data/reference_families.json data/      # dvouhladinová část a kalibrace první nuly, asi minuta
python3 code/missing_zeros.py ZEROS_ROOT B10 3000 1 data/reference_missing_zeros_B10.json  # úplnost (PARI/GP), také C100
python3 -I code/independent_check.py ZEROS_ROOT data/reference_families.json data/reference_subsets.json data/reference_pieces.json data/reference_moduli.json data/reference_characters.json data/reference_pieces_repeat.json data/reference_layers.json data/reference_intermodulation.json data/reference_first_zero_crossing.json data/reference_recomputation.json data/reference_two_level.json data/reference_missing_zeros_B10.json data/reference_families_E100.json
python3 -I code/figures.py data/reference_families.json data/reference_subsets.json data/reference_pieces.json figures/
python3 -I code/figures_moduli.py data/reference_characters.json figures/
python3 -I code/figures_layers.py data/reference_layers.json figures/
python3 -I code/figures_recomputation.py data figures
python3 code/ks_nuly.py --rodiny E100 --jadra 8 --divz 8 --vystup ZEROS_ROOT/set2           # oddíl 4.8: nové bloky E100, asi 128 hodin CPU
python3 -I code/compute_families.py ZEROS_ROOT data/ --families E100                       # E100 = E10 a nové bloky, asi 2 minuty
python3 -I code/layered_recomputation.py ZEROS_ROOT data/ data/ --families E100           # asi 7 minut
python3 -I code/two_level_first_zero.py ZEROS_ROOT data/reference_families_E100.json data/ --families E100
python3 -I code/explicit_weights.py ZEROS_ROOT data/ --families E100                       # asi 5 minut
python3 -I code/e100_trends.py data/ data/                                                 # trendy podle vodiče, sekundy
python3 -I code/figures_e100.py data figures
python3 -I code/export_csv.py data/reference_families.json data/reference_subsets.json data/reference_pieces.json data/ ZEROS_ROOT
```

`ZEROS_ROOT` může obsahovat několik takových sad vedle sebe (každou s vlastním `manifest.json`); rodina se čte ze sady, která ji obsahuje. Výpočet všech rodin trvá asi dvě hodiny na dvou jádrech u menších sad a asi 33 hodin procesorového času u C100, D100, E10 a R10. Znovu vytvořenou sadu lze blok po bloku porovnat s `data/block_content_sha256.csv`.

Výpočty byly vyvinuty a ověřeny v chráněné laboratoři MAYAN ALFA V1008 (interní, verze 34.68e; mayanalfa.com), která není součástí tohoto repozitáře.

## Licence

Text, obrázky a data: CC BY 4.0 (`LICENSE-CC-BY-4.0.md`). Kód: MIT (`LICENSE`).

## Citace

Viz `CITATION.cff`. Tato studie: Hess, D. (2026), Low-lying zeros of 795,381 quadratic Dirichlet L-functions up to conductor 10¹⁰: the ratios conjecture, consecutive families and the first zero — a reproducible numerical study III, Zenodo, [doi:10.5281/zenodo.23991165](https://doi.org/10.5281/zenodo.23991165). Předchozí studie: Hess, D. (2026), Berry saturation and the statistics of the Riemann zeros from t ≈ 10⁴ to 10²¹: a reproducible numerical study, Zenodo, doi:10.5281/zenodo.23815328; Hess, D. (2026), Beyond the diagonal approximation: pair correlation, form factor and resonances of the Riemann zeros from t ≈ 10⁴ to 10²¹ — a reproducible numerical study II, Zenodo, doi:10.5281/zenodo.23928017.
