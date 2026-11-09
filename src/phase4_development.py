"""Calibration/validation-only model-native development. Blind package is absent."""
import hashlib
import itertools
import json
import shutil
import warnings
from pathlib import Path
import numpy as np
import torch
from scipy.stats import binomtest
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .common import ROOT, STATES, read_jsonl, write_jsonl, dump, exact_match, is_refusal, seed_all, fingerprint
from .calibration import utility_correct
from .output_gates import InteractionScore
from .early_evaluate import benchmark_gate
from .phase4_runtime import Phase4Runtime, Phase4Gate, ScopeController, FeatureCapture
from .phase4_common import (load_protocol, native_rows, request, append_jsonl, row_metadata,
    controller_for, objective_quality, fit_five_temperatures)
from .phase4_smoke import smoke


def fit_probe(train, labels, val, layer, C, path):
    train, val = train.astype(np.float64), val.astype(np.float64)
    center = train.mean(0); scale = np.maximum(train.std(0), 1e-5)
    z = (train-center)/scale
    pca = PCA(n_components=32, svd_solver='randomized', random_state=42).fit(z)
    whitening = np.sqrt(np.maximum(pca.explained_variance_, 1e-10))
    transform = pca.components_.T/whitening
    reduced = (z-pca.mean_)@transform
    clf = LogisticRegression(C=C, max_iter=1500, solver='lbfgs', random_state=42, tol=1e-7)
    with warnings.catch_warnings():
        warnings.simplefilter('error', ConvergenceWarning); clf.fit(reduced, labels)
    assert clf.classes_.tolist() == [0, 1, 2, 3]
    linear = transform@clf.coef_.T
    weight = linear/scale[:, None]
    bias = clf.intercept_-(center/scale+pca.mean_)@linear
    expected = ((val-center)/scale-pca.mean_)@transform@clf.coef_.T+clf.intercept_
    error = float(np.max(np.abs(val@weight+bias-expected)))
    assert error < 1e-6 and np.isfinite(weight.astype(np.float16)).all() and np.isfinite(bias.astype(np.float16)).all()
    np.savez(path, weights=weight[None].astype(np.float16), bias=bias.astype(np.float16), layers=[layer])
    return {'folded_max_abs_float64_error': error, 'PCA_explained_variance_ratio': float(pca.explained_variance_ratio_.sum()),
            'logistic_iterations': clf.n_iter_.tolist(), 'offline_dtype': 'float64; stable SVD and linear fit', 'deployment_dtype': 'float16'}


def gate_probabilities(path, runtime, features, layer):
    gate = Phase4Gate(path, {'param_cost': 1., 'abstain_cost': 1., 'param_threshold': .5, 'abstain_threshold': .5},
                      next(runtime.model.model.layers[layer-1].parameters()).device)
    probs, logits = [], []
    with torch.inference_mode():
        for feature in features:
            _, p, z = gate.decide([torch.as_tensor(feature[None], device=gate.weight.device, dtype=torch.float16)])
            probs.append(p[0].float().cpu().numpy()); logits.append(z[0].float().cpu().numpy())
    return np.array(probs), np.array(logits)


def policy_grid(cfg):
    return [dict(zip(['param_cost', 'abstain_cost', 'param_threshold', 'abstain_threshold'], values)) for values in
            itertools.product(cfg['param_costs'], cfg['abstain_costs'], cfg['param_thresholds'], cfg['abstain_thresholds'])]


def route_grid(probabilities, policies):
    # Use the actual online fp16 cost/normalization comparisons, not float64 argmax.
    p = torch.tensor(probabilities, device='cuda:0', dtype=torch.float16)
    cost = torch.tensor([[1., x['param_cost'], 1., x['abstain_cost']] for x in policies], device='cuda:0', dtype=torch.float16)
    adjusted = p[None]*cost[:, None]; adjusted = adjusted/adjusted.sum(-1, keepdim=True)
    winner = adjusted.argmax(-1)
    tparam = torch.tensor([x['param_threshold'] for x in policies], device='cuda:0', dtype=torch.float16)[:, None]
    tabstain = torch.tensor([x['abstain_threshold'] for x in policies], device='cuda:0', dtype=torch.float16)[:, None]
    block = (winner == 1) & (adjusted[:, :, 1] >= tparam)
    refuse = (~block) & (winner == 3) & (adjusted[:, :, 3] >= tabstain)
    route = torch.where(block, 1, torch.where(refuse, 3, torch.where(p[None, :, 2] > p[None, :, 0], 2, 0)))
    return route.cpu().numpy().astype(np.int8)


def quality_grid(routes, base, hard, native, prior):
    state = np.array([STATES.index(r['state']) for r in native]); abstain = state == 3
    bcorrect = np.array([utility_correct(r) for r in base]); bref = np.array([is_refusal(r['answer']) for r in base])
    hcorrect = np.array([utility_correct(r) for r in hard]); href = np.array([is_refusal(r['answer']) for r in hard])
    y = np.where(routes == 1, hcorrect[None], bcorrect[None]); y = np.where(routes == 3, abstain[None], y)
    ref = np.where(routes == 1, href[None], bref[None]); ref = np.where(routes == 3, True, ref)
    rates = np.stack([y[:, state == j].mean(1) for j in range(4)], 1)
    raw = y.mean(1); aligned = (1-prior)/2*(rates[:, 0]+rates[:, 3])+prior/2*(rates[:, 1]+rates[:, 2]); macro = rates.mean(1)
    den = rates[:, 0]+rates[:, 1]
    h = np.divide(2*rates[:, 0]*rates[:, 1], den, out=np.zeros(len(routes)), where=den != 0)
    f1 = 2*(ref & abstain[None]).sum(1)/(ref.sum(1)+abstain.sum())
    return rates, (raw+aligned+macro)/3, h, f1, np.stack([raw, aligned, macro], 1)


def compose(base, hard, route, p, native):
    result = []
    for i, row in enumerate(native):
        item = dict(hard[i] if route[i] == 1 else base[i])
        item.update(row_metadata(row), automatic_route=int(route[i]), route_probabilities=p[i].tolist())
        item['control'] = dict(item.get('control', {}), route=int(route[i]), route_probabilities=p[i].tolist())
        if route[i] == 3: item.update(raw_generated_answer=item['answer'], answer='I do not know', refusal_policy='post_generation_replace')
        result.append(item)
    return result


def develop():
    cfg, lock, facts_path = load_protocol('development'); seeds = seed_all(42)
    out = ROOT/'results'/cfg['study_id']/'development'; out.mkdir(parents=True, exist_ok=True)
    dump(out/'signature.json', {'input_lock': lock, 'seed_state': seeds, 'blind_inputs_present': False,
         'blind_model_calls': 0, 'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}})
    runtime = Phase4Runtime(cfg); smoke(runtime, cfg, out/'technical_smoke')
    facts = read_jsonl(facts_path); closed = {}
    path = out/'closed_book.jsonl'
    for f in facts:
        result = runtime.generate(f['question'])
        item = {'fact_id': f['fact_id'], 'split': f['split'], 'gold_answers': f['gold_answers'], **result,
                'prior_correct': exact_match(result['answer'], f['gold_answers']), 'cost_scope': 'offline native labeling'}
        append_jsonl(path, item); closed[f['fact_id']] = item
        if len(closed) % 20 == 0: print({'stage': 'native_closed_book', 'completed': len(closed), 'total': len(facts)}, flush=True)
    k = sum(x['prior_correct'] for x in closed.values()); prior = k/len(facts)
    ci = binomtest(k, len(facts)).proportion_ci()
    prior_spec = {'p_CB': prior, 'correct': k, 'facts': len(facts), 'ci95_exact': [ci.low, ci.high],
                  'scope': 'unenriched development calibration+validation only; frozen before blind access'}
    dump(out/'native_prior.json', prior_spec)
    native = native_rows(facts, closed); write_jsonl(out/'native_development.jsonl', native)
    cal_index = [i for i, r in enumerate(native) if r['split'] == 'calibration']
    val_index = [i for i, r in enumerate(native) if r['split'] == 'val']
    validation = [native[i] for i in val_index]
    assert len(cal_index) == 800 and len(val_index) == 400
    labels = np.array([STATES.index(native[i]['state']) for i in cal_index])
    assert set(labels) == set(range(4)), 'Calibration misses a state; stop without resizing'
    capture = FeatureCapture(runtime, cfg['layers']); baseline = []; feature_list = []
    try:
        for i, row in enumerate(native):
            result = runtime.generate(row['question'], row['context'], capture=capture)
            item = row_metadata(row) | result | {'method': 'vanilla', 'seed': 42}
            assert result['forward_calls'] == result['generated_tokens']
            baseline.append(item); feature_list.append(capture.array()); append_jsonl(out/'vanilla_development.jsonl', item)
            if (i+1) % 40 == 0: print({'stage': 'native_vanilla_features', 'completed': i+1, 'total': len(native)}, flush=True)
    finally: capture.close()
    features = np.stack(feature_list); assert features.dtype == np.float16 and np.isfinite(features).all()
    np.savez_compressed(out/'development_features.npz', values=features, ids=np.array([r['id'] for r in native]), layers=cfg['layers'])
    base = [baseline[i] for i in val_index]; base_quality = objective_quality(base, prior)
    policies = policy_grid(cfg); state = np.array([STATES.index(r['state']) for r in validation])
    assert set(state) == set(range(4)), 'Validation misses a state; stop without resizing'
    all_trials = []; fits = []; cache = {}; fit_errors = []
    best = None
    for li, layer in enumerate(cfg['layers']):
        layer_fits = []
        for C in cfg['logistic_C']:
            key = fingerprint({'study': cfg['study_id'], 'layer': layer, 'C': C, 'rank': 32})
            artifact = out/(key+'.npz')
            try:
                with threadpool_limits(limits=4): diag = fit_probe(features[cal_index, li], labels, features[val_index, li], layer, C, artifact)
            except (ValueError, AssertionError, ConvergenceWarning) as exc:
                fit_errors.append({'layer': layer, 'C': C, 'error': type(exc).__name__, 'message': str(exc)}); continue
            p, logits = gate_probabilities(artifact, runtime, features[val_index, li], layer)
            routes = route_grid(p, policies); ext_blocks = np.sum((routes == 1) & (state[None] == 0), axis=1)
            safe = ext_blocks <= cfg['max_validation_ext_blocks']
            item = {'layer': layer, 'C': C, 'key': key, 'artifact': artifact.name, 'diagnostics': diag,
                    'routes': routes, 'probabilities': p, 'logits': logits, 'safe': safe, 'ext_blocks': ext_blocks}
            layer_fits.append(item); fits.append(item)
        if not layer_fits: continue
        union = np.any(np.stack([np.any(x['routes'][x['safe']] == 1, axis=0) if x['safe'].any() else np.zeros(len(validation), bool) for x in layer_fits]), axis=0)
        hard = {}
        for scope in ['all_queries', 'last_query']:
            controller = ScopeController(runtime, layer, scope=scope, force_route=1)
            generated = {}
            try:
                for i in np.flatnonzero(union):
                    item = request(runtime, validation[i], 'forced_hard', cfg, controller)
                    generated[validation[i]['id']] = item; append_jsonl(out/f'hard_{layer}_{scope}.jsonl', item)
                    if len(generated) % 40 == 0: print({'stage': 'validation_control_cache', 'layer': layer, 'scope': scope, 'completed': len(generated), 'total': int(union.sum())}, flush=True)
            finally: controller.close()
            hard[scope] = [generated.get(row['id'], base[i]) for i, row in enumerate(validation)]
        cache[layer] = hard
        for fit in layer_fits:
            qa = quality_grid(fit['routes'], base, hard['all_queries'], validation, prior)
            ql = quality_grid(fit['routes'], base, hard['last_query'], validation, prior)
            eligible = fit['safe'].copy()
            for q in [qa, ql]:
                eligible &= q[0][:, 0] >= base_quality['rates']['ext']-cfg['max_validation_ext_accuracy_drop']-1e-12
                eligible &= q[0][:, 2] >= base_quality['rates']['align']-cfg['max_validation_align_accuracy_drop']-1e-12
            for pi, policy in enumerate(policies):
                record = {'key': fit['key'], 'layer': layer, 'C': fit['C'], 'policy': policy,
                          'ext_blocked': int(fit['ext_blocks'][pi]), 'eligible': bool(eligible[pi]),
                          'actual_branches_available': bool(fit['safe'][pi])}
                if fit['safe'][pi]:
                    record.update(mean_utility=float((qa[1][pi]+ql[1][pi])/2), mean_h=float((qa[2][pi]+ql[2][pi])/2),
                                  mean_refusal_f1=float((qa[3][pi]+ql[3][pi])/2),
                                  scopes={scope: {'rates': q[0][pi].tolist(), 'three_utilities': q[4][pi].tolist(),
                                            'h': float(q[2][pi]), 'refusal_f1': float(q[3][pi])}
                                          for scope, q in [('all_queries', qa), ('last_query', ql)]})
                all_trials.append(record)
                if eligible[pi]:
                    objective = (record['mean_utility'], record['mean_h'], record['mean_refusal_f1'], -record['ext_blocked'],
                                 -layer, -fit['C'], tuple(-policy[k] for k in sorted(policy)))
                    if best is None or objective > best[0]: best = (objective, fit, pi, record)
            print({'stage': 'probe_policy_search', 'layer': layer, 'C': fit['C'], 'eligible': int(eligible.sum())}, flush=True)
        dump(out/'policy_trials.json', all_trials); dump(out/'fit_failures.json', fit_errors)
    assert fits, 'No numerically valid probe; no blind evaluation authorized by this failed development'
    fallback = best is None
    if fallback:
        selected_fit = fits[0]; policy = {'param_cost': 1., 'abstain_cost': 1., 'param_threshold': 2., 'abstain_threshold': 2.}
        routes = route_grid(selected_fit['probabilities'], [policy])[0]
        selected_record = {'key': selected_fit['key'], 'layer': selected_fit['layer'], 'C': selected_fit['C'], 'policy': policy,
                           'eligible': False, 'development_failed': True, 'reason': 'No eligible policy; registered keep-only fallback'}
    else:
        _, selected_fit, pi, selected_record = best; policy = policies[pi]; routes = selected_fit['routes'][pi]
    layer = selected_fit['layer']; shutil.copyfile(out/selected_fit['artifact'], out/'frozen_gate.npz')
    gate = Phase4Gate(out/'frozen_gate.npz', policy, next(runtime.model.model.layers[layer-1].parameters()).device)
    val_outputs = {'vanilla': base}
    for method, scope in [('early_gate', 'all_queries'), ('last_query', 'last_query')]:
        expected = compose(base, cache[layer][scope], routes, selected_fit['probabilities'], validation)
        controller = controller_for(runtime, method, cfg, layer, gate); actual = []; differences = []
        try:
            for i, row in enumerate(validation):
                item = request(runtime, row, method, cfg, controller); actual.append(item)
                if item['answer'] != expected[i]['answer'] or item['control']['route'] != int(routes[i]) or item['generated_token_ids'] != expected[i]['generated_token_ids']:
                    differences.append({'id': row['id'], 'cached_answer': expected[i]['answer'], 'online_answer': item['answer'],
                                        'cached_route': int(routes[i]), 'online_route': item['control']['route']})
                append_jsonl(out/('validation_'+method+'.jsonl'), item)
                if (i+1) % 40 == 0: print({'stage': 'complete_online_validation', 'method': method, 'completed': i+1}, flush=True)
        finally: controller.close()
        dump(out/(method+'_composition_differences.json'), differences)
        assert not differences, 'Cached/online action mismatch; stop before blind'
        val_outputs[method] = actual
    encoder = InteractionScore(); scored = []
    for row in validation:
        cb = closed[row['fact_id']]
        score, _ = encoder.score(cb['answer'], row['context'], cb['first_token']['confidence']); scored.append(score)
    thresholds = [-1., *np.quantile(scored, cfg['interaction_threshold_quantiles']).tolist(), 2.]
    interaction_trials = []
    for threshold in thresholds:
        rows = [row_metadata(row) | dict(closed[row['fact_id']] if scored[i] > threshold else base[i]) | {'method': 'interaction', 'quality_source': 'validation offline cached branches'} for i, row in enumerate(validation)]
        quality = objective_quality(rows, prior); interaction_trials.append({'threshold': threshold, 'quality': quality, 'rows': rows})
    chosen_interaction = max(interaction_trials, key=lambda x: (x['quality']['mean_utility'], x['quality']['h_score'], -x['threshold']))
    threshold = chosen_interaction['threshold']; val_outputs['interaction'] = chosen_interaction['rows']
    dump(out/'interaction_threshold_trials.json', [{k: v for k, v in x.items() if k != 'rows'} for x in interaction_trials])
    for method in ['cad', 'static_span', 'static_hard', 'oracle']:
        controller = controller_for(runtime, method, cfg, layer, gate); rows = []
        try:
            for i, row in enumerate(validation):
                item = request(runtime, row, method, cfg, controller); rows.append(item)
                append_jsonl(out/('validation_'+method+'.jsonl'), item)
                if (i+1) % 40 == 0: print({'stage': 'validation_baseline', 'method': method, 'completed': i+1}, flush=True)
        finally:
            if controller is not None: controller.close()
        val_outputs[method] = rows
    shared = fit_five_temperatures(val_outputs['early_gate']+val_outputs['last_query'])
    temperatures = {m: shared if m in ['early_gate', 'last_query'] else fit_five_temperatures(val_outputs[m]) for m in cfg['methods']}
    val_quality = {m: objective_quality(rows, prior) for m, rows in val_outputs.items()}
    benchmark = benchmark_gate(gate)
    freeze = {'complete': True, 'study_id': cfg['study_id'], 'model': cfg['model'], 'revision': cfg['revision'],
        'input_lock': lock, 'blind_inputs_present': False, 'blind_model_calls': 0, 'seed_state': seeds,
        'selected': selected_record, 'layer': layer, 'policy': policy, 'development_fallback': fallback,
        'weights_sha256': hashlib.sha256((out/'frozen_gate.npz').read_bytes()).hexdigest(),
        'native_prior': prior_spec, 'interaction_threshold': threshold, 'temperatures': temperatures,
        'numerical_policy': json.loads((ROOT/'configs/phase4_qwen7b_numerics.json').read_text()),
        'validation_quality': val_quality, 'gate_benchmark': benchmark,
        'calibration_sample_ids': [native[i]['id'] for i in cal_index], 'validation_sample_ids': [r['id'] for r in validation],
        'source_sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')},
        'all_scope_differences': 'same gate, route, policy, temperatures, fp16 and one-prefill KV; only query mask scope differs'}
    dump(out/'freeze.json', freeze)
    print({'development_complete': True, 'layer': layer, 'native_p_CB': prior, 'blind_model_calls': 0,
           'validation_all': val_quality['early_gate'], 'validation_last': val_quality['last_query']}, flush=True)


if __name__ == '__main__': develop()
