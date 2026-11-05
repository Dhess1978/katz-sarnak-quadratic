# Low-lying zeros of 795,381 quadratic Dirichlet L-functions up to conductor 10¹⁰: the ratios conjecture, consecutive families and the first zero (study III)

**David Hess (MAYAN ALFA V1008 LAB v.34.68e) · 5.11.2026** · [Česká verze](README.cs.md)

Third study of the series, after *Berry saturation and the statistics of the Riemann zeros* (Zenodo, [doi:10.5281/zenodo.23815328](https://doi.org/10.5281/zenodo.23815328)) and *Beyond the diagonal approximation* (study II; Zenodo, [doi:10.5281/zenodo.23928017](https://doi.org/10.5281/zenodo.23928017)). It moves from the Riemann zeta function to the family of quadratic Dirichlet L-functions L(s, χ_D) of Section 6 of Berry and Keating (1999): eleven families of fundamental discriminants from |D| ≈ 10³ to 10¹⁰, 795,381 L-functions, the low zeros computed with PARI/GP on two computers (the 69,452 L-functions computed on both have identical zeros). The one-level density of the unfolded zeros x = θ_D(γ)/π is compared with the Katz–Sarnak limit USp, with the finite matrix USp(2N) and with the ratios conjecture of Conrey, Farmer and Zirnbauer in the form of Conrey and Snaith (2007), evaluated for every discriminant with no free parameter.

| result | value |
|---|---|
| one-level density, χ² per bin (bins of 0.05, x ≤ 4, Poisson error of the observed count), ratios conjecture, 11 families / expected for independent L-functions | 0.54 – 1.19 / about 0.95 |
| the same against the USp limit (families of at least 6,000 L-functions) | 5 – 850 |
| density in [0, ½], 607,941 L-functions with 10⁶ < \|D\| ≤ 2·10⁶: measured / ratios conjecture / USp | 0.6082 ± 0.0014 / 0.6084 / 0.411 |
| the same at \|D\| ≈ 10¹⁰ (12,153 L-functions) | 0.527 ± 0.009 / 0.527 / 0.411 (USp 12.6 errors away) |
| low-frequency power (cosine modes k = 1 … 7, u < 1) of bands of 6,074 consecutive discriminants, relative to random samples of the same size, \|D\| ≈ 5·10⁴ / 1.5·10⁶ / 10⁸ | 0.23 / 0.36 / 0.48 (5.9 / 10.2 / 4.3 errors) |
| the same at high frequencies (k = 32 … 79) | no difference |
| stratification of random samples by D mod 840: reduction of the low-frequency power | 14 – 16 % in the first samples (a fifth to a quarter of the deficit); with four to five times more samples 18 % in B10, not significant in C100 and D100 |
| stratification by (sign, D mod M) for 19 moduli (3 … 840 and 131) and by the sign alone, with an independent repetition | does not reproduce the effect: C100 and D100 as random samples, B10 only mod 840 (z = −5.1) |
| the line through random samples and consecutive bands: exact remaining variance W₁₀₀ of the sums of χ_D(p), p ≤ 100 (no zeros) → low-frequency power of the zeros of all 60 stratified subsets | right direction; size at the edge: χ² = 82.05 for 60 points with the joint errors (p = 0.03), 42 of 60 deviations positive |
| profile r_p of consecutive bands (Δ ≈ 10⁴) in three families, 582 primes up to 4·10⁶ | the same function of p: 0.04–0.05 for p ≤ 100, 0.84–0.90 up to 10⁵, about 1 above |
| first zero, USp(2N) with N = ½ ln(\|D\|/π): KS statistic / critical value | 1.6 – 53 |
| first zero, USp(2N_eff) with N_eff from the ratios conjecture, odd characters | KS below the critical value at every conductor |
| the same, even characters: first zero higher by | about 0.094/ln(\|D\|/π) (14.8 errors in the family of 607,941) |
| the same, even characters: crossing of the distribution with USp(2N_eff), C100 | 0.508 ± 0.021; follows the end a of the interval [0, a] on which N_eff is fixed (slope 1.13 for a = 0.3 … 0.7), so not a property of the zeros |
| exact expectation of the low-frequency power of a subset design without draws (layer 0); test statistic above it for random samples, B10 / C100 / D100 | 2.7 / 3.5 / 1.9 %, positive for all 30 designs: a bias of the two lowest bins (84 % of it in B10), not noise |
| triples of consecutive discriminants / random samples | 0.972 in B10 (4.7 errors), 0.975 in C100 (4.3) |
| correlation of the cosine modes 1 and 7 of one L-function, B10 / C100 / D100 (Poisson zeros: 0) | 0.58 / 0.47 / 0.39 |
| increase of the explained share of mode 7 by χ_D(n) of 16 composite n < e⁴, B10 / C100 / D100 (same features permuted) | 0.188 / 0.111 / 0.043 (at most 0.0003) |
| recomputation of every drawn number of Sections 4.3–4.5 by layers (Section 4.7), same designs: error of the low-frequency power / check of layer 0 | 1.6 – 4.6 times smaller (median 3.7) / mean z² 1.13 for 84 designs; the original values agree within the joint errors |
| the same: stratification by D mod 840 against random samples, B10 / C100 / D100 | z = −26.6 / −13.5 / −2.9; 15 / 16 / 3 of 19 moduli beyond 3 errors, but small (17 / 9.5 / 1.6 % of the random value) |
| the line through random samples and bands with the values by layers | χ² = 841.8 for 60: rejected in size; exactly 0.89 / 0.56 / 0.39 of the reduction of the line |
| weights of χ_D(p) in the contribution of one L-function, slope fitted / explicit formula: diagonal terms / with the dual term of the ratios conjecture | 0.52 – 0.60 / 0.995 – 0.999 |
| mean first zero of even characters minus USp(2N_eff(a)), N_eff fixed on [0, a], C100, a = 0.3 / 0.5 / 0.7 | +0.0153 / +0.0072 / −0.0012 (± 0.0005): depends on the calibration; the shape (KS) deviates in B10 and C100 at every a |
| zeros missed by the search (step / 8) in 3,000 L-functions, B10 / C100 | one pair in one L-function / none |
| family E100 at \|D\| = 10¹⁰ (Section 4.8), 121,557 L-functions: χ² per bin against the ratios conjecture / without A and Δ / USp | 0.759 / 16.5 / 66.7; density in [0, ½] 0.5291 ± 0.0030 (ratios conjecture 0.5267, USp 0.4105) |
| one consecutive stretch / random samples, exact, B10 / C100 / D100 / E100 | 0.204 / 0.391 / 0.606 / 0.622; share of the reduction predicted by single primes 89 / 76 / 65 / 42 % |
| E100, mean first zero minus USp(2N_eff), even / odd; shape F(1) − USp(2N_eff) of even characters | +0.0016 / −0.0009 (± 0.0012), 2.3 errors below 0.094/l; −0.0029 ± 0.0015 against −0.0069 from B10–D100: falls faster than 1/l |

The ratios conjecture and the Katz–Sarnak limit are known theory; the contribution is a systematic, reproducible test on a large set of zeros with null models and an independent check, the identification of the low χ² of consecutive families as an effect of the choice of the family (complete periods of χ_D(p) for primes below the length of the stretch, seen at low frequencies through the explicit formula) and its prediction from the character sums without zeros, the even–odd difference of the first zero, the exact expectation of subset designs without random draws and the coupling of cosine modes through products of characters of small primes. Section 6 of the article separates what is new, as far as we have been able to find, from what is confirmed. Nothing here bears on the generalised Riemann hypothesis.

## Contents

| folder | what |
|---|---|
| `paper/` | the article: `katz_sarnak_quadratic_en.pdf`, LaTeX source `main.tex`, Markdown and DOCX versions; Czech version in `paper/cs/` |
| `figures/` | 10 figures, English (`_en`) and Czech (`_cz`), PNG 300 dpi and PDF (vector) |
| `data/` | `reference_families.json`, `reference_subsets.json`, `reference_pieces.json`, `reference_pieces_repeat.json`, `reference_moduli.json`, `reference_characters.json`, `reference_layers.json`, `reference_intermodulation.json`, `reference_first_zero_crossing.json`, `reference_recomputation.json`, `reference_explicit_weights.json`, `reference_two_level.json`, `reference_missing_zeros_B10.json`, `reference_missing_zeros_C100.json` (all numbers of Section 4), CSV exports per family, bin, subset, piece (with the error of every point), modulus, character sum, prime of the band profile, design of layer 0, mode of the intermodulation and crossing of the first zero, the recomputation by layers, the weights of the explicit formula, the two-level part of the first zero and the missing zeros, `reference_families_E100.json`, `reference_recomputation_E100.json`, `reference_two_level_E100.json`, `reference_explicit_weights_E100.json` and `reference_e100_trends.json` (Section 4.8, the family E100; E100 is also in the CSV exports, with the trends over the conductor in `e100_trends.csv`), `block_content_sha256.csv` (SHA-256 of the decompressed content of every block of zeros), the manifests of the five sets of zeros |
| `code/` | `ks_nuly.py` (generator of the zeros, PARI/GP via cypari2), `ks_kontrola.py` (completeness check without PARI), `read_zeros.py`, `compute_families.py`, `pieces_repeat.py`, `compute_moduli.py`, `character_sums.py`, `exact_layers.py`, `intermodulation.py`, `first_zero_crossing.py`, `layered_recomputation.py`, `explicit_weights.py`, `two_level_first_zero.py`, `missing_zeros.py`, `independent_check.py`, `figures.py`, `figures_moduli.py`, `figures_layers.py`, `figures_recomputation.py`, `e100_trends.py`, `figures_e100.py`, `export_csv.py` |

## Reproducing

The zeros are not included (3,806,316 zeros in the eleven families, 3,452,121 distinct, and 471,278 in the 900 new blocks of E100 of Section 4.8; about 35 MB compressed); they are regenerated by `code/ks_nuly.py`, the script that produced them (its messages and comments are in Czech; `--help` lists the options). The manifests record the SHA-256 of the generator revision that produced each set; the revisions up to version 34.68c differ only in the list of families and in a runtime estimate; the current revision adds the option `--divz` (step of the search = mean spacing / divz, default 32, Section 4.7 of the paper) and with `--divz 8` reproduces the zeros of the oldest set (families A and B) identically.

```
pip install -r requirements.txt          # cypari2 (PARI/GP) only for ks_nuly.py, missing_zeros.py and checks 3 and 16 of independent_check.py
python3 code/ks_nuly.py --test
python3 code/ks_nuly.py --rodiny A,B,B10,C10,C100,D10,D100,E,E10,R,R10 --jadra 8 --divz 8 --vystup ZEROS_ROOT/set1   # the published data: step = mean spacing / 8
python3 code/ks_kontrola.py ZEROS_ROOT/set1                                                 # completeness and format, without PARI
python3 -I code/compute_families.py ZEROS_ROOT data/                                        # about 5 minutes
python3 -I code/pieces_repeat.py ZEROS_ROOT data/                                           # test with pieces, 400 subsets per point with errors, about 4 minutes
python3 -I code/compute_moduli.py ZEROS_ROOT data/reference_moduli.json                     # Section 4.4, about 8 minutes
python3 -I code/character_sums.py ZEROS_ROOT data/reference_moduli.json data/reference_characters.json   # about 20 minutes
python3 -I code/exact_layers.py ZEROS_ROOT data/                                           # Section 4.6, about 40 minutes, 4 GB RAM
python3 -I code/intermodulation.py ZEROS_ROOT data/                                        # intermodulation and filters, about 2 minutes
python3 -I code/first_zero_crossing.py ZEROS_ROOT data/reference_families.json data/       # crossing of the first zero with the calibration test
python3 -I code/layered_recomputation.py ZEROS_ROOT data/ data/                           # Section 4.7 by layers, about 25 minutes, 4 GB RAM
python3 -I code/explicit_weights.py ZEROS_ROOT data/                                       # explicit formula with the dual term, about 20 minutes, 4 GB RAM
python3 -I code/two_level_first_zero.py ZEROS_ROOT data/reference_families.json data/      # two-level part and calibration of the first zero, about a minute
python3 code/missing_zeros.py ZEROS_ROOT B10 3000 1 data/reference_missing_zeros_B10.json  # completeness (PARI/GP), also C100
python3 -I code/independent_check.py ZEROS_ROOT data/reference_families.json data/reference_subsets.json data/reference_pieces.json data/reference_moduli.json data/reference_characters.json data/reference_pieces_repeat.json data/reference_layers.json data/reference_intermodulation.json data/reference_first_zero_crossing.json data/reference_recomputation.json data/reference_two_level.json data/reference_missing_zeros_B10.json data/reference_families_E100.json
python3 -I code/figures.py data/reference_families.json data/reference_subsets.json data/reference_pieces.json figures/
python3 -I code/figures_moduli.py data/reference_characters.json figures/
python3 -I code/figures_layers.py data/reference_layers.json figures/
python3 -I code/figures_recomputation.py data figures
python3 code/ks_nuly.py --rodiny E100 --jadra 8 --divz 8 --vystup ZEROS_ROOT/set2           # Section 4.8: the new blocks of E100, about 128 CPU hours
python3 -I code/compute_families.py ZEROS_ROOT data/ --families E100                       # E100 = E10 and the new blocks, about 2 minutes
python3 -I code/layered_recomputation.py ZEROS_ROOT data/ data/ --families E100           # about 7 minutes
python3 -I code/two_level_first_zero.py ZEROS_ROOT data/reference_families_E100.json data/ --families E100
python3 -I code/explicit_weights.py ZEROS_ROOT data/ --families E100                       # about 5 minutes
python3 -I code/e100_trends.py data/ data/                                                 # trends over the conductor, seconds
python3 -I code/figures_e100.py data figures
python3 -I code/export_csv.py data/reference_families.json data/reference_subsets.json data/reference_pieces.json data/ ZEROS_ROOT
```

`ZEROS_ROOT` may hold several such sets side by side (each with its own `manifest.json`); a family is read from whichever set contains it. Generating all families takes about two hours on two cores for the smaller sets and about 33 CPU hours for C100, D100, E10 and R10. A regenerated set can be compared with `data/block_content_sha256.csv` block by block.

The computations were developed and verified in the protected MAYAN ALFA V1008 laboratory (internal, version 34.68e; mayanalfa.com), which is not part of this repository.

## Licence

Text, figures and data: CC BY 4.0 (`LICENSE-CC-BY-4.0.md`). Code: MIT (`LICENSE`).

## Citation

See `CITATION.cff`. Previous studies: Hess, D. (2026), Berry saturation and the statistics of the Riemann zeros from t ≈ 10⁴ to 10²¹: a reproducible numerical study, Zenodo, doi:10.5281/zenodo.23815328; Hess, D. (2026), Beyond the diagonal approximation: pair correlation, form factor and resonances of the Riemann zeros from t ≈ 10⁴ to 10²¹ — a reproducible numerical study II, Zenodo, doi:10.5281/zenodo.23928017.
