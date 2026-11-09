"""Cloud-only source-fit CAST/ContextFocus supplement. Original artifacts read-only."""
import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, dump, read_jsonl, seed_all
from .model_runtime import Runtime, SYSTEM
from .phase4_runtime import Phase4Runtime, synchronize
from .phase4_common import append_jsonl, objective_quality, fit_five_temperatures, attach_probability
from .internal_control_baselines_v1 import (
    pairwise_pca, contextfocus_direction, condition_score, fit_condition_threshold,
    ResidualController, SteeringSpec, FirstTokenTopK)

CF_SYSTEMS = [
    'Use the supplied passage as the sole evidence for your answer.',
    'Ground your answer entirely in the passage included with this question.',
    'For this task, derive the answer only from the supplied text.',
    'Treat the attached passage as the reference for answering this question.',
    'Respond using the information stated in the accompanying passage.',
    'Limit the answer to what the given passage supports.',
    'Base your response on this passage rather than recalled facts.',
    'The supplied text is the evidence source for this answer.',
    'Answer this question by following the accompanying text.',
    'Rely exclusively on the attached text to determine the answer.',
    'Give the answer supported by the passage presented here.',
    'Consult only the provided passage when forming your response.',
    'Use contextual evidence in the supplied passage to answer.',
    'In this exercise, the passage determines the answer.',
    'Keep your answer grounded in the text that follows.',
    'Select the answer from information in the accompanying text.',
    'Let the provided passage guide the factual content of your answer.',
    'Answer according to the supplied evidence, setting aside recalled alternatives.',
    'Use the accompanying reference text to resolve the question.',
    'Restrict your evidence for this response to the provided text.'
]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def macro_summary(rows):
    return objective_quality(rows, prior=0.1)


def choose_config(candidates):
    # Macro first, H second; exact ties prefer lower layer, alpha, lexical method.
    return sorted(candidates, key=lambda c: (-c['validation']['macro_utility'],
        -c['validation']['h_score'], c['spec']['layer'], c['spec']['alpha'], c['spec']['method']))[0]


class LayerCapture:
    def __init__(self, model, layers):
        self.positions = [-1]
        self.values = {}
        self.handles = []
        for layer in layers:
            self.handles.append(model.model.layers[layer-1].register_forward_hook(self.hook(layer)))

    def hook(self, layer):
        def run(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            # Mean suffix representations for refusal/compliance; last token otherwise.
            self.values[layer] = h[0, self.positions, :].float().mean(0).detach().cpu().numpy()
        return run

    def close(self):
        for handle in self.handles:
            handle.remove()


def custom_prompt(runtime, question, context=None, system=SYSTEM, suffix=None):
    user = 'Question: ' + question if context is None else '<context>\n' + context + '\n</context>\nQuestion: ' + question
    messages = ([{'role': 'system', 'content': system}] if system else []) + [{'role': 'user', 'content': user}]
    text = runtime.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    start = len(text)
    if suffix is not None:
        text += suffix
    encoded = runtime.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    if len(encoded['input_ids']) > runtime.max_input:
        raise ValueError('Contrast prompt exceeds 512; exclude paired offline contrast, never truncate')
    positions = [i for i, (a, b) in enumerate(encoded['offset_mapping']) if b > start and b > a] if suffix is not None else [-1]
    if not positions:
        raise ValueError('Empty suffix tokens')
    inputs = {'input_ids': torch.tensor([encoded['input_ids']], device='cuda:0'),
              'attention_mask': torch.ones((1, len(encoded['input_ids'])), dtype=torch.long, device='cuda:0')}
    return inputs, positions


@torch.inference_mode()
def fit_directions(runtime, rows, layers, out):
    assert all(r['split'] == 'calibration' for r in rows)
    capture = LayerCapture(runtime.model, layers)
    calls = runtime.calls
    audit = []
    prompt_rng = np.random.default_rng(42)
    def extract(inputs, positions, label):
        capture.positions, capture.values = positions, {}
        runtime.model(**inputs, use_cache=False, num_logits_to_keep=1)
        result = {l: capture.values[l].copy() for l in layers}
        assert all(np.isfinite(x).all() for x in result.values())
        audit.append({'contrast': label, 'prompt_tokens': int(inputs['input_ids'].shape[1]), 'forward_calls': 1})
        return result
    byfact = defaultdict(list)
    for r in rows:
        byfact[r['fact_id']].append(r)
    contexts = {}
    cb, cf_positive, cf_negative, refusal, compliance = {}, {}, {}, {}, {}
    try:
        for i, r in enumerate(rows):
            inputs, _ = runtime.encode(r['question'], r['context'])
            contexts[r['id']] = extract(inputs, [-1], 'condition_context:' + r['id'])
            if (i+1) % 100 == 0:
                print({'offline_context_prefills': i+1}, flush=True)
        for i, (fid, pair) in enumerate(sorted(byfact.items())):
            assert len(pair) == 2
            good = next(r for r in pair if r['context_correct'])
            wrong = next(r for r in pair if not r['context_correct'])
            if good['prior_correct']:
                cb[fid] = extract(*custom_prompt(runtime, good['question']), 'closed_book:' + fid)
            # Negative ContextFocus representation: question only, no system.
            negative = custom_prompt(runtime, good['question'], system=None)
            for r in pair:
                try:
                    variant = int(prompt_rng.integers(len(CF_SYSTEMS)))
                    positive = custom_prompt(runtime, r['question'], r['context'], system=CF_SYSTEMS[variant])
                except ValueError:
                    audit.append({'excluded_offline_pair': 'contextfocus:' + r['id'], 'reason': 'length'})
                    continue
                if fid not in cf_negative:
                    cf_negative[fid] = extract(*negative, 'contextfocus_negative:' + fid)
                cf_positive[r['id']] = extract(*positive, 'contextfocus_positive:' + r['id'])
                audit[-1]['system_variant'] = variant
            # Domain adaptation of refusal/compliance, teacher forced, no sampling.
            try:
                pos = custom_prompt(runtime, wrong['question'], wrong['context'], suffix='I do not know.')
                neg = custom_prompt(runtime, wrong['question'], wrong['context'], suffix=wrong['designated_answer'])
            except ValueError:
                audit.append({'excluded_offline_pair': 'refusal:' + fid, 'reason': 'length'})
            else:
                refusal[fid] = extract(*pos, 'refusal_suffix:' + fid)
                compliance[fid] = extract(*neg, 'compliance_suffix:' + fid)
            if (i+1) % 50 == 0:
                print({'offline_fact_contrasts': i+1, 'total_facts': len(byfact)}, flush=True)
    finally:
        capture.close()
    weights, metadata = {}, {}
    for layer in layers:
        good = [next(r for r in pair if r['context_correct']) for pair in byfact.values()]
        bad = [next(r for r in pair if not r['context_correct']) for pair in byfact.values()]
        cond = pairwise_pca([contexts[r['id']][layer] for r in bad], [contexts[r['id']][layer] for r in good])
        # Fit threshold on CPU fp16 with the same tanh/rank-one formula as deployment.
        hidden = torch.tensor(np.stack([contexts[r['id']][layer] for r in rows]), dtype=torch.float16)
        scores = condition_score(hidden, torch.tensor(cond, dtype=torch.float16)).float().numpy()
        threshold = fit_condition_threshold(scores, [not r['context_correct'] for r in rows])
        known_bad = [r for r in bad if r['fact_id'] in cb]
        rag = pairwise_pca([cb[r['fact_id']][layer] for r in known_bad],
                           [contexts[r['id']][layer] for r in known_bad])
        ref = pairwise_pca([refusal[f][layer] for f in refusal], [compliance[f][layer] for f in refusal])
        cf_ids = [r for r in rows if r['id'] in cf_positive]
        cf = contextfocus_direction([cf_positive[r['id']][layer] for r in cf_ids],
                                     [cf_negative[r['fact_id']][layer] for r in cf_ids])
        for name, vec in [('condition', cond), ('cast_rag', rag), ('cast_refusal', ref), ('contextfocus', cf)]:
            weights[f'{name}_{layer}'] = vec
        metadata[str(layer)] = dict(threshold, condition_pairs=len(byfact), recovery_pairs=len(known_bad),
                                     refusal_pairs=len(refusal), contextfocus_pairs=len(cf_ids))
    assert runtime.calls-calls == sum(r.get('forward_calls', 0) for r in audit)
    np.savez_compressed(out / 'directions.npz', **weights)
    dump(out / 'direction_fit.json', {'layers': metadata, 'offline_forward_calls': runtime.calls-calls,
        'source_split': 'calibration', 'float64_reason': 'stable offline pairwise PCA/mean estimation; model and deployment remain fp16',
        'offline_records': audit, 'contextfocus_systems': CF_SYSTEMS, 'refusal_suffix': 'I do not know.',
        'compliance_suffix': 'source designated context answer, teacher forced, calibration only',
        'cast_token_boundary': 'all_prefill_plus_decode', 'contextfocus_token_boundary': 'prefill_last_plus_decode'})
    return weights, metadata


def evaluate(runtime, rows, spec, weights, directory, temperatures=None):
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f'{spec.method}_l{spec.layer}_a{spec.alpha:g}.jsonl'
    assert not path.exists(), 'No implicit repeat of validation/quality outputs'
    controller = ResidualController(runtime.model, spec, weights[f'{spec.method}_{spec.layer}'],
        weights[f'condition_{spec.layer}'] if spec.method.startswith('cast') else None)
    observer = FirstTokenTopK(runtime.model)
    outputs = []
    try:
        for i, r in enumerate(rows):
            observer.reset()
            result = runtime.generate(r['question'], r['context'], controller=controller)
            assert result['forward_calls'] == result['generated_tokens']
            audit = controller.audit()
            assert audit['residual_calls'] == result['generated_tokens']
            assert audit['condition_calls'] == int(spec.method.startswith('cast'))
            assert audit['query_lengths'] == [result['prompt_tokens']] + [1]*(result['generated_tokens']-1)
            assert observer.values is not None and len(observer.values['token_ids']) == 10
            item = {k: r[k] for k in ['id', 'fact_id', 'state', 'gold_answers', 'split', 'split_id', 'prior_correct', 'component_id'] if k in r}
            item.update(result, method=spec.method, seed=42, control=audit, first_token_top10=observer.values)
            if temperatures is not None:
                item = attach_probability(item, temperatures)
            append_jsonl(path, item)
            outputs.append(item)
            if (i+1) % 50 == 0:
                print({'method': spec.method, 'layer': spec.layer, 'alpha': spec.alpha,
                       'split': r['split'], 'completed': i+1, 'total': len(rows)}, flush=True)
    finally:
        controller.close()
        observer.close()
    return outputs, path


def technical_check(runtime, row, weights, layer, out):
    """Explicitly counted validation-input hook check, separate from grid/quality."""
    before = runtime.calls
    observer = FirstTokenTopK(runtime.model)
    checks = []
    base = None
    try:
        for name, alpha in [('vanilla', 0.), ('cast_zero', 0.), ('cast_forced', 2.), ('contextfocus', 2.)]:
            controller = None
            if name != 'vanilla':
                method = 'contextfocus' if name == 'contextfocus' else 'cast_rag'
                spec = SteeringSpec(method, layer, alpha, threshold=-2.)
                controller = ResidualController(runtime.model, spec, weights[f'{method}_{layer}'],
                    weights[f'condition_{layer}'] if method.startswith('cast') else None)
            try:
                observer.reset()
                result = runtime.generate(row['question'], row['context'], controller=controller)
                signature = {'answer': result['answer'], 'generated_tokens': result['generated_tokens'], 'top10': observer.values}
                if base is None:
                    base = signature
                if name == 'cast_zero':
                    assert signature == base, 'Real-model zero-alpha identity failed'
                audit = controller.audit() if controller else None
                if controller:
                    assert audit['attention_edits'] == 0
                    assert audit['residual_edits'] == result['generated_tokens'] * (alpha > 0)
                checks.append({'name': name, 'forward_calls': result['forward_calls'], 'control': audit,
                    'zero_identity': signature == base, 'first_top10_changed': signature['top10'] != base['top10'],
                    'cache_lengths': result.get('cache_lengths'), 'cache_devices': result.get('cache_devices')})
            finally:
                if controller:
                    controller.close()
    finally:
        observer.close()
    assert runtime.calls-before == sum(x['forward_calls'] for x in checks)
    dump(out / 'technical_checks.json', {'source_split': 'val', 'used_for_selection': False,
                                        'forward_calls': runtime.calls-before, 'requests': checks})


def main(model):
    assert Path('/kaggle/working').is_dir(), 'No local backbone calls'
    cfg = json.loads((ROOT / 'configs/reviewer_baselines_v1.json').read_text())
    folder = ROOT / 'data/processed/reviewer_baselines_v1' / model
    lock = json.loads((folder / 'input_lock.json').read_text())
    for name, digest in lock['files'].items():
        assert sha(folder / name) == digest
    assert sha(ROOT / 'configs/reviewer_baselines_v1.json') == lock['configuration_sha256']
    out = ROOT / 'results/reviewer_baselines_v1' / model
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'entry_lock.json').exists()
    state = seed_all(42)
    dump(out / 'entry_lock.json', {'seed_state': state, 'input_lock': lock, 'nqswap_generation': False})
    if model == 'qwen3b':
        runtime = Runtime(max_input=512, max_new=12)
    else:
        runtime_cfg = json.loads((ROOT / 'configs/phase4_qwen7b_v1.json').read_text())
        runtime_cfg.update(max_input_tokens=512, max_new_tokens=12)
        runtime = Phase4Runtime(runtime_cfg)
    layers = cfg['layers'][model]
    development = read_jsonl(folder / 'development.jsonl')
    cal = [r for r in development if r['split'] == 'calibration']
    val = [r for r in development if r['split'] == 'val']
    weights, fitted = fit_directions(runtime, cal, layers, out)
    technical_check(runtime, val[0], weights, layers[0], out)
    candidates, val_outputs = [], {}
    for method in ['cast_rag', 'cast_refusal', 'contextfocus']:
        for layer in layers:
            for alpha in cfg['alphas']:
                spec = SteeringSpec(method=method, layer=layer, alpha=alpha,
                    threshold=fitted[str(layer)]['threshold'], comparator=fitted[str(layer)]['comparator'])
                records, path = evaluate(runtime, val, spec, weights, out / 'validation')
                candidate = {'spec': asdict(spec), 'validation': macro_summary(records), 'path': path.relative_to(ROOT).as_posix()}
                candidates.append(candidate)
                val_outputs[str(spec)] = records
    choices = {method: choose_config([c for c in candidates if c['spec']['method'].startswith('cast') == (method == 'cast')])
               for method in ['cast', 'contextfocus']}
    for name, choice in choices.items():
        spec = SteeringSpec(**choice['spec'])
        choice['calibration_folds'] = fit_five_temperatures(val_outputs[str(spec)])
        choice['temperatures'] = [x['temperature'] for x in choice['calibration_folds']]
    freeze = {'model': model, 'choices': choices, 'candidates': candidates, 'seed': 42,
        'directions_sha256': sha(out / 'directions.npz'), 'input_lock_sha256': sha(folder / 'input_lock.json'),
        'blind_read_before_freeze': False, 'nqswap_generation': False,
        'baseline_scope': 'task-domain adaptations; compact source-only selection, author mechanisms preserved; no safety-task replication claim'}
    dump(out / 'freeze.json', freeze)
    # Read old blind only AFTER unique selection and temperature freeze are on disk.
    blind = read_jsonl(folder / 'blind.jsonl')
    assert not {r['fact_id'] for r in blind} & {r['fact_id'] for r in development}
    for name, choice in choices.items():
        records, path = evaluate(runtime, blind, SteeringSpec(**choice['spec']), weights, out / 'blind', choice['temperatures'])
        dump(out / f'{name}_blind_summary.json', macro_summary(records))
    synchronize()
    dump(out / 'completion.json', {'complete': True, 'model': model, 'validation_configurations': 18,
        'blind_methods': 2, 'blind_contexts': len(blind), 'total_backbone_forward_calls': runtime.calls,
        'freeze_sha256': sha(out / 'freeze.json'), 'all_backbone_parameters_frozen_fp16':
            all(not p.requires_grad and p.dtype == torch.float16 for p in runtime.model.parameters()),
        'extra_online_model_calls': 0, 'attention_edits': 0, 'manual_refusal_replacements': 0,
        'quality_budget': '12 tokens or EOS, matching old PopQA caches; not fixed-budget timing evidence'})
    import sklearn
    dump(out / 'environment.json', {'torch': torch.__version__, 'numpy': np.__version__,
        'sklearn': sklearn.__version__, 'seed_state': state,
        'model': model, 'attention_backend': 'eager', 'model_dtype': 'float16',
        'devices': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'device_map': getattr(runtime.model, 'hf_device_map', {}),
        'model_manifest': 'results/kaggle_model_manifest.json' if model == 'qwen3b' else 'results/phase4_model_manifest.json'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['qwen3b', 'qwen7b'], required=True)
    main(parser.parse_args().model)
