"""Locked independent NQ confirmation: source transfer and domain calibration."""
import argparse
import itertools
import json
import os
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax
from threadpoolctl import threadpool_limits
from .common import ROOT, STATES, dump, read_jsonl, write_jsonl, seed_all, exact_match
from .calibration import utility_correct
from .early_gate import route_probabilities
from .early_controller import FoldedGate
from .active_block_closure import ActiveFoldedGate
from .phase4_runtime import FeatureCapture
from .phase4_development import fit_probe
from .phase4_common import append_jsonl, objective_quality, fit_five_temperatures, attach_probability
from .output_gates import InteractionScore
from .nqswap_runtime_v2 import (sha, profile, runtime_for, native_rows, generate,
                               request, replay, control_for, environment)


def score(rows, prior):
    return objective_quality(rows, prior)


def gate_outputs(runtime, path, features, policy):
    gate = FoldedGate(path, policy, 'cuda:0')
    probs, logits, routes = [], [], []
    with torch.inference_mode():
        for x in features:
            r, p, z = gate.decide([torch.as_tensor(x[None], device='cuda:0', dtype=torch.float16)])
            routes.append(int(r.item())); probs.append(p[0].float().cpu().numpy()); logits.append(z[0].float().cpu().numpy())
    del gate
    return np.array(routes), np.array(probs), np.array(logits)


def routes_from_prob(prob, policy):
    # Execute the same fp16 cost/threshold operations on the deployment GPU.
    with torch.inference_mode():
        p = torch.as_tensor(prob, dtype=torch.float16, device='cuda:0')
        q = p*torch.tensor([1., policy['param_cost'], 1., policy['abstain_cost']], dtype=torch.float16, device='cuda:0')
        q = q/q.sum(-1, keepdim=True)
        winner = q.argmax(-1)
        block = (winner == 1) & (q[:, 1] >= policy['param_threshold'])
        refuse = ~block & (winner == 3) & (q[:, 3] >= policy['abstain_threshold'])
        return torch.where(block, 1, torch.where(refuse, 3, torch.where(p[:, 2] > p[:, 0], 2, 0))).cpu().numpy()


def choose_policy(prob, base, hard, prior):
    trials = []
    ext = np.array([r['state'] == 'ext' for r in base])
    base_ext = float(np.mean([exact_match(r['answer'], r['gold_answers']) for r in base if r['state'] == 'ext']))
    for pc, ac, pt, at in itertools.product([1., 2., 4., 8.], [.5, 1., 2.], [.2, .35, .5], [.2, .35, .5]):
        policy = dict(param_cost=pc, abstain_cost=ac, param_threshold=pt, abstain_threshold=at)
        routes = routes_from_prob(prob, policy)
        mask = np.isin(routes, [1, 3])
        quality = score(replay(base, hard, mask, 'calibration_policy_trial'), prior)
        trials.append(dict(policy=policy, quality=quality, ext_intercepts=int(mask[ext].sum()),
                           eligible=bool(quality['rates']['ext'] >= base_ext-.01-1e-12)))
    eligible = [r for r in trials if r['eligible']]
    if eligible:
        selected = sorted(eligible, key=lambda r: (-r['quality']['macro_utility'], -r['quality']['h_score'],
            r['ext_intercepts'], *r['policy'].values()))[0]
    else:
        policy = dict(param_cost=1., abstain_cost=1., param_threshold=2., abstain_threshold=2.)
        selected = dict(policy=policy, quality=score(base, prior), eligible=True, fallback_no_block=True)
    return selected, trials


def run_rows(runtime, model, rows, method, out, encoder, features=False):
    path = out / f'{method}.jsonl'
    assert not path.exists(), 'No implicit rerun of a locked quality branch'
    capture = FeatureCapture(runtime, [profile(model)['layer']]) if features else None
    records, captured = [], []
    try:
        for i, r in enumerate(rows):
            item = request(runtime, r, model, method, encoder, capture=capture)
            append_jsonl(path, item); records.append(item)
            if capture is not None:
                captured.append(capture.array()[0])
            if (i+1) % 50 == 0:
                print(dict(stage=r['split'], method=method, completed=i+1, total=len(rows)), flush=True)
    finally:
        if capture:
            capture.close()
    return records, np.asarray(captured, dtype=np.float16) if features else None


def closed_book(runtime, facts, out):
    result = {}
    for i, f in enumerate(facts):
        r = generate(runtime, f['question'])
        r.update(fact_id=f['fact_id'], split=f['split'], prior_correct=exact_match(r['answer'], f['org_answer']), seed=42)
        result[f['fact_id']] = r
        append_jsonl(out, r)
        if (i+1) % 50 == 0:
            print(dict(stage='offline_closed_book', completed=i+1, total=len(facts)), flush=True)
    return result


def attach(rows, temps, name):
    return [dict(attach_probability(r, temps), method=name) for r in rows]


def technical(runtime, model, rows, policies, paths, predictions, out):
    # Actual auto-path generation uses only source/new frozen probe, never labels.
    selected = []
    for state in STATES:
        selected.extend([r for r in rows if r['state'] == state][:2])
    audits = []
    for track in ['zero_shot', 'calibrated']:
        gate = ActiveFoldedGate(paths[track], policies[track], 'cuda:0')
        lookup = {r['id']: r for r in predictions[track]}
        for row in selected:
            r = request(runtime, row, model, 'active_block', gate=gate)
            expected = lookup[row['id']]
            assert r['answer'] == expected['answer']
            assert (r['control']['route'] == 1) == expected['intercepted']
            assert r['control']['gate_calls'] == 1
            audits.append(dict(r, track=track, validation_technical_only=True))
        del gate
    write_jsonl(out / 'actual_automatic_validation_checks.jsonl', audits)
    return audits


def main(model):
    assert Path('/kaggle/working').is_dir(), 'Cloud-only model access'
    os.environ['SLEA_NQ_STRONG_CAST'] = '1'
    cfg = json.loads((ROOT / 'configs/nqswap_protocol_v2.json').read_text())
    lockpath = ROOT / 'data/processed/nqswap_v2/sample_lock.json'
    lock = json.loads(lockpath.read_text()); assert lock['valid_cohort']
    for name, digest in lock['files'].items():
        assert sha(lockpath.parent / name) == digest
    cfglock = json.loads((ROOT / 'data/processed/nqswap_v2/protocol_lock.json').read_text())
    assert sha(ROOT / 'configs/nqswap_protocol_v2.json') == cfglock['configuration_sha256']
    assert sha(ROOT / 'docs/nqswap_preregistration_v2.md') == cfglock['preregistration_sha256']
    out = ROOT / 'results/nqswap_v2' / model
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'entry_lock.json').exists()
    seeds = seed_all(42)
    dump(out / 'entry_lock.json', dict(input_lock=lock, protocol_lock=cfglock, seed_state=seeds,
        sample_lock_sha256=sha(lockpath), blind_semantic_access_before_freeze=False))
    runtime = runtime_for(model, 32)
    dump(out / 'environment.json', environment(runtime, model))
    encoder = InteractionScore()
    p = profile(model)
    dev_facts = read_jsonl(lockpath.parent / 'calibration_facts.jsonl') + read_jsonl(lockpath.parent / 'val_facts.jsonl')
    cb = closed_book(runtime, dev_facts, out / 'closed_book_development.jsonl')
    prior = float(np.mean([r['prior_correct'] for r in cb.values()]))
    dump(out / 'native_prior_lock.json', dict(p_cb=prior, n_facts=400, measured_on='unenriched calibration+validation before blind access'))
    native = native_rows(dev_facts, cb)
    write_jsonl(out / 'native_development.jsonl', native)
    cal = [r for r in native if r['split'] == 'calibration']; val = [r for r in native if r['split'] == 'val']
    (out / 'calibration').mkdir(); (out / 'validation').mkdir()
    cal_v, xcal = run_rows(runtime, model, cal, 'vanilla', out / 'calibration', encoder, True)
    validation = {}
    validation['vanilla'], xval = run_rows(runtime, model, val, 'vanilla', out / 'validation', encoder, True)
    for method in ['all_hard', 'last_hard', 'cad', 'interaction', 'cast', 'contextfocus']:
        validation[method], _ = run_rows(runtime, model, val, method, out / 'validation', encoder)
    np.savez_compressed(out / 'development_features.npz', calibration=xcal, validation=xval,
        calibration_ids=[r['id'] for r in cal], validation_ids=[r['id'] for r in val], layer=p['layer'])
    probe = out / 'calibrated_probe.npz'
    with threadpool_limits(limits=4):
        fit = fit_probe(xcal, np.array([STATES.index(r['state']) for r in cal]), xval, p['layer'], 1., probe)
    paths = dict(zero_shot=ROOT / p['gate_weights'], calibrated=probe)
    default = dict(param_cost=1., abstain_cost=1., param_threshold=.5, abstain_threshold=.5)
    _, prob_c, _ = gate_outputs(runtime, probe, xval, default)
    chosen, trials = choose_policy(prob_c, validation['vanilla'], validation['all_hard'], prior)
    policies = dict(zero_shot=p['gate_policy'], calibrated=chosen['policy'])
    routes, logits = {}, {}
    for track in paths:
        routes[track], _, logits[track] = gate_outputs(runtime, paths[track], xval, policies[track])
    assert np.array_equal(routes['calibrated'], routes_from_prob(prob_c, policies['calibrated']))
    predicted, temperatures, folds = {}, {}, {}
    for track in paths:
        predicted[track] = replay(validation['vanilla'], validation['all_hard'], np.isin(routes[track], [1, 3]), track+'_all')
        predicted[track+'_last'] = replay(validation['vanilla'], validation['last_hard'], np.isin(routes[track], [1, 3]), track+'_last')
        for name in [track, track+'_last']:
            if track == 'zero_shot':
                temperatures[name] = p['active_last_temperatures'] if name.endswith('_last') else p['active_temperatures']
            else:
                folds[name] = fit_five_temperatures(predicted[name])
                temperatures[name] = [f['temperature'] for f in folds[name]]
    freeze = dict(model=model, layer=p['layer'], fit=fit, selected_policy=chosen, policy_trials=trials,
        gate_policies=policies, gate_weights_sha256={t: sha(path) for t, path in paths.items()},
        temperatures=temperatures, calibration_folds=folds, p_cb=prior, seed=42,
        source_profiles_sha256=sha(ROOT / 'configs/nqswap_source_profiles_v2.json'),
        blind_read_before_freeze=False, old_weights_modified=False, sample_lock_sha256=sha(lockpath))
    freeze['cast_source_freeze_sha256'] = sha(ROOT / 'results/cast_sensitivity_v2' / model / 'nq_cast_freeze.json')
    dump(out / 'freeze.json', freeze)
    actual = technical(runtime, model, val, policies, paths, predicted, out)
    for name, rr in predicted.items():
        write_jsonl(out / 'validation' / (name+'.jsonl'), attach(rr, temperatures[name], name))
    # First semantic access to independent blind inputs, after all fits/freezes.
    blind_facts = read_jsonl(lockpath.parent / 'blind_facts.jsonl')
    blind_cb = closed_book(runtime, blind_facts, out / 'closed_book_blind.jsonl')
    blind = native_rows(blind_facts, blind_cb)
    write_jsonl(out / 'native_blind.jsonl', blind)
    (out / 'blind').mkdir()
    branches = {}
    branches['vanilla'], xb = run_rows(runtime, model, blind, 'vanilla', out / 'blind', encoder, True)
    for method in ['all_hard', 'last_hard', 'cad', 'interaction', 'cast', 'contextfocus']:
        branches[method], _ = run_rows(runtime, model, blind, method, out / 'blind', encoder)
    np.savez_compressed(out / 'blind_features.npz', values=xb, ids=[r['id'] for r in blind], layer=p['layer'])
    summary = {}
    for track in paths:
        route, probability, z = gate_outputs(runtime, paths[track], xb, policies[track])
        np.savez_compressed(out / (track+'_blind_gate_outputs.npz'), routes=route, probabilities=probability, logits=z, ids=[r['id'] for r in blind])
        for suffix, branch in [('', 'all_hard'), ('_last', 'last_hard')]:
            name = track+suffix
            rr = replay(branches['vanilla'], branches[branch], np.isin(route, [1, 3]), name)
            rr = attach(rr, temperatures[name], name)
            write_jsonl(out / 'blind' / (name+'.jsonl'), rr)
            summary[name] = score(rr, prior)
    for name in ['vanilla', 'cad', 'interaction', 'cast', 'contextfocus']:
        temps = p['baseline_temperatures'][name]
        rr = attach(branches[name], temps, name)
        write_jsonl(out / 'blind' / (name+'_frozen_confidence.jsonl'), rr)
        summary[name] = score(rr, prior)
    dump(out / 'point_summary.json', summary)
    dump(out / 'completion.json', dict(complete=True, model=model, independent_facts=700, blind_facts=300,
        blind_contexts=600, input_lock_sha256=sha(lockpath), freeze_sha256=sha(out / 'freeze.json'),
        total_model_forward_calls=runtime.calls, offline_cb_requests=700, quality_branch_requests=6800,
        actual_automatic_validation_requests=len(actual), first_token_top10_recorded=True,
        main_method='Unified ActiveBlock-All', explicit_refusal_replacements=0, extra_arbitration_model_calls=0,
        second_gate_added_blind_generation_branches=0, cross_model_holm6='computed locally after both jobs complete'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--model', choices=['qwen3b', 'qwen7b'], required=True)
    main(parser.parse_args().model)
