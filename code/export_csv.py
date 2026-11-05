#!/usr/bin/env python3
"""CSV exports of study III from the outputs of compute_families.py; with ZEROS_ROOT also the SHA-256 of the decompressed content of every block of zeros
(so that a regenerated set can be compared without publishing the zeros). If reference_moduli.json and reference_characters.json (compute_moduli.py, character_sums.py)
lie in the folder of reference_families.json, also moduli.csv, moduli_subsets.csv, character_sums.csv and band_profile.csv (section 4.4); likewise layers.csv,
intermodulation.csv and first_zero_crossing.csv from reference_layers.json, reference_intermodulation.json and reference_first_zero_crossing.json (sections 4.5-4.6).
Section 4.7: recomputation.csv, recomputation_line.csv, explicit_weights.csv, two_level.csv and missing_zeros.csv from reference_recomputation.json,
reference_explicit_weights.json, reference_two_level.json and reference_missing_zeros_*.json. Section 4.8: the family E100 (files *_E100.json) in the same tables
and e100_trends.csv from reference_e100_trends.json.
usage: python3 -I export_csv.py reference_families.json reference_subsets.json reference_pieces.json OUTPUT_FOLDER [ZEROS_ROOT]"""
import sys, os, json, csv
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
RF, RS, RP, OUT = sys.argv[1:5]; ROOT = sys.argv[5] if len(sys.argv) > 5 else None; os.makedirs(OUT, exist_ok=True)
F = json.load(open(RF)); S = json.load(open(RS)); P = json.load(open(RP))
if os.path.exists(RF.replace('.json', '_E100.json')): F['families'].update(json.load(open(RF.replace('.json', '_E100.json')))['families'])   # section 4.8
def w(name, header, rows):
    with open(os.path.join(OUT, name), 'w', newline='') as fh: c = csv.writer(fh); c.writerow(header); c.writerows(rows)
    print(name, len(rows))
r = lambda x, d=6: None if x is None else round(x, d)
rows = []
for f, R in F['families'].items():
    for s, q in R.items():
        b = q['bootstrap']
        rows.append([f, s, q['n'], r(q['mean_ln_D'], 4), r(q['k025']['chi2_ratios'], 4), r(q['k05']['chi2_ratios'], 4), r(q['k1']['chi2_ratios'], 4), r(q['k05']['independent'], 4), r(b['k05']['q05'], 4), r(b['k05']['q95'], 4), r(b['k05']['p_below'], 3),
                     r(q['chi2_005_without_AD'], 4), r(q['chi2_005_usp'], 3), r(q['chi2_005_usp2n'], 3), r(q['density_0_05'], 5), r(q['density_0_05_se'], 5), r(q['density_0_05_ratios'], 5), r(q['density_0_05_usp'], 5),
                     r(q['zeros_to_4'], 5), r(q['zeros_to_4_ratios'], 5), r(q['x1_mean'], 5), r(q['x1_se'], 5), r(q['x1_usp2n_eff'], 5), r(q['N_eff'], 4), r(q['ks_usp2n_eff'], 5), r(q['x1_usp2n'], 5), r(q['ks_usp2n'], 4), r(q['ks_usp'], 4), r(q['ks_critical'], 5)])
w('family_summary.csv', ['family', 'sign', 'n_L', 'mean_ln_D', 'chi2_bin0025_ratios', 'chi2_bin005_ratios', 'chi2_bin01_ratios', 'chi2_bin005_independent_L', 'bootstrap_q05', 'bootstrap_q95', 'p_below',
                         'chi2_bin005_without_AD', 'chi2_bin005_USp', 'chi2_bin005_USp2N', 'density_0_05', 'density_0_05_se', 'density_0_05_ratios', 'density_0_05_USp', 'zeros_to_x4', 'zeros_to_x4_ratios',
                         'x1_mean', 'x1_se', 'x1_USp2Neff', 'N_eff', 'KS_USp2Neff', 'x1_USp2N', 'KS_USp2N', 'KS_USp', 'KS_critical'], rows)
rows = []
for f, R in F['families'].items():
    for s, q in R.items():
        for b, (h, p) in enumerate(zip(q['hist_0025'], q['prediction_0025'])): rows.append([f, s, round((b + 0.5)*F['xf'], 4), h, q['n'], p])
w('family_histograms.csv', ['family', 'sign', 'x_center', 'count', 'n_L', 'prediction_ratios'], rows)
rows = []
for f, R in F['families'].items():
    for s, q in R.items():
        for x, a, b in zip(F['sg'], q['x1_cdf'], q['x1_cdf_usp2n_eff']): rows.append([f, s, x, a, b])
w('first_zero_cdf.csv', ['family', 'sign', 'x', 'F_data', 'F_USp2Neff'], rows)
rows = []; rows2 = []
for f, R in S['families'].items():
    cnt = {}
    for q in R['subsets']:
        cnt[q['type']] = cnt.get(q['type'], 0) + 1; rows.append([f, q['type'], cnt[q['type']], q['n'], q['D_abs'][0], q['D_abs'][1], r(q['chi2_005'], 5), r(q['chi2_01'], 5), r(q['independent'], 5), r(q['low_k1_7'], 6), r(q['high_k32_79'], 6)])
    for typ in sorted(cnt):
        A = [q['power'] for q in R['subsets'] if q['type'] == typ]
        for k in range(len(A[0])): rows2.append([f, typ, k, round(k/8, 4), round(sum(a[k] for a in A)/len(A), 8), len(A)])
w('subsets.csv', ['family', 'type', 'i', 'n_L', 'D_abs_min', 'D_abs_max', 'chi2_bin005', 'chi2_bin01', 'chi2_independent_L', 'power_k1_7', 'power_k32_79'], rows)
w('subset_spectra.csv', ['family', 'type', 'k', 'u', 'mean_power', 'n_subsets'], rows2)
rows = []
for f, R in P['families'].items():
    for m, q in sorted(R['m'].items(), key=lambda kv: int(kv[0])): rows.append([f, int(m), round(q['stretch_D'], 2), round(q['u_c'], 4), round(q['chi2_005'], 5), round(q['chi2_005_se'], 5)] + [round(x, 6) for x in q['bands']])
w('pieces.csv', ['family', 'm_pieces', 'stretch_D_abs', 'u_c', 'chi2_bin005', 'chi2_se', 'power_k0', 'power_k1_3', 'power_k4_7', 'power_k8_15', 'power_k16_31', 'power_k32_79'], rows)
RP2 = os.path.join(os.path.dirname(os.path.abspath(RP)), 'reference_pieces_repeat.json')   # test with pieces with the error of every point (pieces_repeat.py)
if os.path.exists(RP2):
    P2 = json.load(open(RP2)); rows = []
    for run, key in (('first_run', 'first_run'), ('repetition', 'repetition')):
        for f, Rf in P2[key]['families'].items():
            for m, q in sorted(Rf['m'].items(), key=lambda kv: int(kv[0])): rows.append([run, f, int(m), q['n_subsets'], round(q['stretch_D'], 2), round(q['u_c'], 4), round(q['power_k1_7'], 6), round(q['power_k1_7_se'], 6), round(q['power_k1_7_sd'], 6)])
    for sd_, Q in sorted(P2['seeds_B10'].items()):
        for m, q in sorted(Q.items(), key=lambda kv: int(kv[0])): rows.append(['seed_' + sd_, 'B10', int(m), P2['repetition']['subsets'], '', '', round(q['power_k1_7'], 6), round(q['power_k1_7_se'], 6), ''])
    w('pieces_repeat.csv', ['run', 'family', 'm_pieces', 'n_subsets', 'stretch_D_abs', 'u_c', 'power_k1_7', 'power_k1_7_se', 'power_k1_7_sd'], rows)
RM = os.path.join(os.path.dirname(os.path.abspath(RF)), 'reference_moduli.json'); RC = os.path.join(os.path.dirname(os.path.abspath(RF)), 'reference_characters.json')
if os.path.exists(RM) and os.path.exists(RC):
    M = json.load(open(RM)); C = json.load(open(RC)); SU = C['summary']; rows = []; rows2 = []
    for f, F_ in M['families'].items():
        for typ, q in F_['types'].items():
            t = SU['families'][f]['types'][typ]
            rows.append([f, typ, q['n'], r(q['low_mean'], 7), r(q['low_se'], 7), r(t.get('z'), 3), r(t.get('W100'), 5), r(t.get('W1000'), 5), r(t.get('pred_W100'), 7), r(t.get('zp_W100'), 3), r(t.get('WE'), 5), r(t.get('pred_WE'), 7), r(t.get('zp_WE'), 3), r(t.get('W100_draws'), 5)])
            for ps, v in q['passes'].items():
                for i, (lo, ch) in enumerate(zip(v, q['chi'][ps])): rows2.append([f, typ, ps, i + 1, lo, ch])
    w('moduli.csv', ['family', 'type', 'n_subsets', 'power_k1_7_mean', 'power_k1_7_se', 'z_vs_random', 'W100_exact', 'W1000_exact', 'line_prediction_W100', 'z_vs_line_W100', 'WE_exact', 'line_prediction_WE', 'z_vs_line_WE', 'W100_from_draws'], rows)
    w('moduli_subsets.csv', ['family', 'type', 'pass', 'i', 'power_k1_7', 'chi2_bin005'], rows2)
    rows = []; SP = None
    for f, F_ in C['sums']['families'].items():
        for typ, q in F_['types'].items():
            SP = SP or [p for p, _ in q['small_primes']]; rows.append([f, typ, F_['n'], r(F_['random_over_theory'], 5), r(q['W100'], 5), r(q['W1000'], 5)] + [r(v, 5) for _, v in q['small_primes']])
    w('character_sums.csv', ['family', 'type', 'n_family', 'random_over_finite_population', 'W100', 'W1000'] + ['variance_ratio_p%d' % p for p in SP], rows)
    rows = []
    for f, F_ in C['profile']['families'].items():
        for p, v in zip(C['profile']['primes'], F_['r']): rows.append([f, p, v])
    w('band_profile.csv', ['family', 'p', 'r_p'], rows)
DD = os.path.dirname(os.path.abspath(RF)); RL_, RI_, RX_ = (os.path.join(DD, x) for x in ('reference_layers.json', 'reference_intermodulation.json', 'reference_first_zero_crossing.json'))
if os.path.exists(RL_):
    L_ = json.load(open(RL_)); rows = []
    for f, F_ in L_['families'].items():
        for k, d in F_['designs'].items(): rows.append([f, k, d['size'], r(d['layer0']), r(d['estimate'][0]), r(d['estimate'][1]), r(d['draws_only'][0]), r(d['draws_only'][1]), r(d['difference'][0]), r(d['difference'][1]), d['draws'][0] + d['draws'][1]])
        rows.append([f, 'stretch_all_starts', L_['subset_size'], r(F_['stretch']['layer0']), r(F_['stretch']['own']), '', '', '', '', '', F_['stretch']['starts']])
        for P_, q in sorted(F_['balanced'].items(), key=lambda kv: int(kv[0])): rows.append([f, 'balanced_p%s' % P_, L_['subset_size'], r(q['layer0_regression']), r(q['estimate'][0]), r(q['estimate'][1]), r(q['mean']), r(q['se']), r(q['difference'][0]), r(q['difference'][1]), q['subsets']])
        for P_, q in sorted(F_['regression'].items(), key=lambda kv: int(kv[0])): rows.append([f, 'regression_p%s' % P_, '', r(q['layer0']), '', '', '', '', '', '', ''])
    w('layers.csv', ['family', 'design', 'size', 'layer0', 'estimate', 'estimate_se', 'draws_only', 'draws_only_se', 'difference_test_minus_layer0', 'difference_se', 'subsets_drawn'], rows)
if os.path.exists(RI_):
    I_ = json.load(open(RI_)); rows = []
    for f, F_ in I_['families'].items():
        for W_ in F_['windows']:
            for k in range(7): rows.append([f, r(W_['ln_n'][0], 4), r(W_['ln_n'][1], 4), W_['features'], k + 1, r(F_['base_explained'][k]), r(W_['gain'][k]), r(W_['null_mean'][k], 7), r(W_['null_sd'][k], 7)])
    w('intermodulation.csv', ['family', 'ln_n_from', 'ln_n_to', 'composites', 'mode_k', 'explained_share_primes_300', 'gain_with_composites', 'null_gain_mean', 'null_gain_sd'], rows)
if os.path.exists(RX_):
    X_ = json.load(open(RX_)); rows = []
    for f, F_ in X_['families'].items():
        e = F_['even']; rows.append([f, F_['plus']['n'], r(F_['plus']['N_eff']), r(e['crossing']), r(e['crossing_se']), r(e['crossing_even_odd']), r(e['crossing_even_odd_se']), r(e['max']), r(e['max_x'], 3), r(e['min']), r(e['min_x'], 3), r(e['min_x_se']), r(e['reenter_ks_band'], 3), r(e['reenter_se']), r(e['value_at_1_5'][0], 7), r(e['value_at_1_5'][1], 7), r(F_['plus']['ks_critical'])])
    w('first_zero_crossing.csv', ['family', 'n_even', 'N_eff_even', 'crossing', 'crossing_se', 'crossing_even_odd', 'crossing_even_odd_se', 'max', 'max_x', 'min', 'min_x', 'min_x_se', 'reenter_ks_band', 'reenter_se', 'dF_at_1_5', 'dF_at_1_5_se', 'ks_critical'], rows)
RH_, RE_, RT_ = (os.path.join(DD, x) for x in ('reference_recomputation.json', 'reference_explicit_weights.json', 'reference_two_level.json'))
ST_ = ('low', 'high', 'chi05', 'chi01')
J_ = lambda p: [json.load(open(x)) for x in (p, p.replace('.json', '_E100.json')) if os.path.exists(x)]   # section 4.8: the family E100 in the same tables
if os.path.exists(RH_):                                                      # section 4.7: recomputation by layers
    rows = []; rows2 = []
    for H_ in J_(RH_):
     for f, F_ in H_['families'].items():
         for k, d in F_['designs'].items():
             for i, st in enumerate(ST_): rows.append([f, k, d['size'], d['draws'], st, r(d['layer0'][st]), r(d['estimate'][st][0]), r(d['estimate'][st][1]), r(d['draws_only'][st][0]), r(d['draws_only'][st][1]), r(d['error_ratio'][st], 4), r(d['layer0_check_z'][st], 3)])
         for i, st in enumerate(ST_):
             rows.append([f, 'bands', H_['subset_size'], F_['bands']['n'], st, r(F_['bands']['layer0_statistic'][st][0]), r(F_['bands']['own'][st][0]), r(F_['bands']['own'][st][1]), '', '', '', ''])
             rows.append([f, 'stretch_all_starts', H_['subset_size'], F_['stretch']['starts'], st, '', r(F_['stretch']['own'][i]), '', '', '', '', ''])
         for k, q in H_['line_layer0_exact'][f]['rows'].items(): rows2.append(['layer0_exact', f, k, r(q['exact']), r(q['line']), r(q['reduction_exact']), r(q['reduction_line'])])
    w('recomputation.csv', ['family', 'design', 'size', 'draws_or_count', 'statistic', 'layer0', 'estimate', 'estimate_se', 'draws_only', 'draws_only_se', 'error_ratio', 'layer0_check_z'], rows)
    for H_ in J_(RH_):
        for v, q in H_['line'].items(): rows2.append(['line_' + v, 'all' if len(H_['families']) > 1 else next(iter(H_['families'])), q['n'], r(q['chi2'], 3), r(q['p'], 6), q['positive'], r(q['mean_z'], 3)])
    w('recomputation_line.csv', ['row', 'family', 'design_or_n', 'exact_or_chi2', 'line_or_p', 'reduction_exact_or_positive', 'reduction_line_or_mean_z'], rows2)
if os.path.exists(RE_):
    rows = []
    for E_ in J_(RE_):
     for f, F_ in E_['families'].items():
         R_ = F_['regression']; S_ = F_['windows_summary']
         rows += [[f, 'slope_diagonal', '', r(R_['slope_chi'][0]), r(R_['slope_chi'][1])], [f, 'slope_with_dual', '', r(R_['slope_chi_with_dual'][0]), r(R_['slope_chi_with_dual'][1])],
                  [f, 'windows_chi2_diagonal', S_['n'], r(S_['chi2_diagonal'], 4), ''], [f, 'windows_chi2_with_dual', S_['n'], r(S_['chi2_with_dual'], 4), '']]
         for k in range(7):
             rows += [[f, 'slope_diagonal_mode', k + 1, r(R_['slope_chi_by_mode'][k][0]), r(R_['slope_chi_by_mode'][k][1])], [f, 'slope_with_dual_mode', k + 1, r(R_['slope_chi_with_dual_by_mode'][k][0]), r(R_['slope_chi_with_dual_by_mode'][k][1])],
                      [f, 'explained_share_diagonal_mode', k + 1, r(F_['explained_without_fit_diagonal'][k]), ''], [f, 'explained_share_with_dual_mode', k + 1, r(F_['explained_without_fit'][k]), '']]
         for M_, q in F_['stratifications'].items(): rows += [[f, 'reduction_exact', M_, r(q['reduction_exact']), ''], [f, 'reduction_predicted_with_dual', M_, r(q['reduction_predicted']), ''], [f, 'reduction_predicted_diagonal', M_, r(q['reduction_predicted_diagonal']), '']]
         T_ = F_['stretch']; rows += [[f, 'reduction_exact', 'stretch', r(T_['reduction_exact']), ''], [f, 'reduction_predicted_with_dual', 'stretch', r(T_['reduction_sum']), ''], [f, 'reduction_predicted_diagonal', 'stretch', r(T_['reduction_sum_diagonal']), '']]
    w('explicit_weights.csv', ['family', 'quantity', 'mode_or_design', 'value', 'se'], rows)
if os.path.exists(RT_):
    rows = []
    for T_ in J_(RT_):
     for f, F_ in T_['families'].items():
         for sg in ('plus', 'minus'):
             q = F_[sg]
             for a_, d in q['rows'].items(): rows.append([f, sg, 'count', a_, q['n'], r(q['N_eff'], 4), r(d['EN'][0]), r(d['EN'][1]), r(d['ENN1'][0]), r(d['ENN1'][1]), r(d['F'][0]), r(d['F'][1]), r(d['model']['EN']), r(d['model']['ENN1']), r(d['model']['F']), '', ''])
             for c in q['calibration']: rows.append([f, sg, 'calibration', c['a'], q['n'], r(c['N_eff'], 4), r(c['x1_data']), r(c['x1_se']), '', '', '', '', r(c['x1_model']), '', '', r(c['ks']), r(c['ks_critical'])])
    w('two_level.csv', ['family', 'sign', 'row', 'a', 'n', 'N_eff', 'EN_or_x1_data', 'EN_se_or_x1_se', 'ENN1', 'ENN1_se', 'F', 'F_se', 'model_EN_or_x1', 'model_ENN1', 'model_F', 'ks', 'ks_critical'], rows)
RTR_ = os.path.join(DD, 'reference_e100_trends.json')
if os.path.exists(RTR_):                                                     # section 4.8: trends over the conductor with the family E100
    Q_ = json.load(open(RTR_)); rows = []
    for f, c in Q_['conductor'].items():
        for k in ('random', 'mod840'): rows.append([f, r(c['l'], 4), c['n'], 'layered_' + k, r(c[k][0]), r(c[k][1])])
        for k in ('stretch_all_starts', 'ratio_stretch_random', 'reduction_mod840', 'z_mod840', 'layer0_reduction_ratio_to_line', 'W100_all_starts'): rows.append([f, r(c['l'], 4), c['n'], k, r(c[k]), ''])
        for k in ('slope_diagonal', 'slope_dual'): rows.append([f, r(c['l'], 4), c['n'], k, r(c[k][0]), r(c[k][1])])
    for f, c in Q_['shift'].items():
        for sg in ('plus', 'minus'): rows += [[f, r(c['l'], 4), c[sg]['n'], 'x1_minus_USp2Neff_' + sg, r(c[sg]['d']), r(c[sg]['se'])], [f, r(c['l'], 4), c[sg]['n'], 'ks_over_critical_' + sg, r(c[sg]['ks']/c[sg]['ks_critical'], 4), '']]
    for f, c in Q_['dF1'].items():
        for k in ('plus', 'minus', 'plus_05', 'plus_07'): rows.append([f, r(c['l'], 4), '', 'F1_minus_USp2Neff_' + k, r(c[k][0]), r(c[k][1])])
    for nm in ('shift_fit', 'dF1_fit'):
        for v in ('without_E100', 'with_E100'):
            for law, q in Q_[nm][v].items(): rows.append([nm + '_' + v, '', '', law + ('' if law != 'free' else '_power_%s' % q['a']), r(q['c']), r(q.get('se', 0) or 0)])
    w('e100_trends.csv', ['family_or_fit', 'l', 'n', 'quantity', 'value', 'se'], rows)
rows = []
for f in ('B10', 'C100'):
    q_ = os.path.join(DD, 'reference_missing_zeros_%s.json' % f)
    if os.path.exists(q_):
        q = json.load(open(q_)); rows.append([f, q['n_family'], q['sample'], q['seed'], q['divz_data'], q['divz_check'], q['same_as_data'], q['missing_L'], q['missing_zeros'], ' '.join('%.4f' % x for x in q['missing_x'])])
if rows: w('missing_zeros.csv', ['family', 'n_family', 'sample', 'seed', 'divz_data', 'divz_check', 'same_as_data', 'missing_L', 'missing_zeros', 'missing_x'], rows)
if ROOT:
    import read_zeros as RZ
    w('block_content_sha256.csv', ['family', 'block', 'n_L', 'n_zeros', 'sha256_of_decompressed_content'], RZ.content_sha256(ROOT))
