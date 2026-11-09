"""Validation-only high-strength CAST and explicitly labeled multi-layer ablation."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from .common import ROOT, dump, read_jsonl, write_jsonl, seed_all, is_refusal
from .nqswap_runtime_v2 import runtime_for, profile, request, generate, sha, environment
from .internal_control_baselines_v1 import ResidualController, SteeringSpec, condition_score
from .reviewer_baseline_job_v1 import LayerCapture, custom_prompt
from .phase4_common import objective_quality, append_jsonl, fit_five_temperatures
from .active_block_closure import ActiveFoldedGate
from .output_gates import InteractionScore


def gram_pc(positive, negative):
    delta = (np.asarray(positive, np.float64)-np.asarray(negative, np.float64))/2
    eigen, vectors = np.linalg.eigh(delta@delta.T)
    if eigen[-1] <= 0:
        raise ValueError('Degenerate paired residual direction')
    direction = delta.T@vectors[:, -1]
    direction /= np.linalg.norm(direction)
    if np.mean(delta@direction) < 0:
        direction = -direction
    return direction.astype(np.float32)


class MultiCAST:
    """Condition at l*, distinct source-calibrated direction at every l*..L."""
    def __init__(self, model, first, vectors, condition, threshold, comparator, alpha):
        self.first, self.threshold, self.comparator, self.alpha = first, threshold, comparator, alpha
        self.condition = torch.as_tensor(condition, dtype=torch.float16, device=next(model.model.layers[first-1].parameters()).device)
        self.vectors = {l: torch.as_tensor(v, dtype=torch.float16, device=next(model.model.layers[l-1].parameters()).device)
                        for l, v in vectors.items()}
        self.handles = [model.model.layers[l-1].register_forward_hook(self.hook(l)) for l in sorted(vectors)]
        self.reset()

    def reset(self, span=None):
        self.calls = defaultdict(int); self.edits = defaultdict(int)
        self.condition_calls = 0; self.active = False; self.similarity = None

    def hook(self, layer):
        def apply(module, args, output):
            h = output[0] if isinstance(output, tuple) else output
            self.calls[layer] += 1
            if layer == self.first and self.calls[layer] == 1:
                self.similarity = float(condition_score(h[0, -1], self.condition))
                self.active = self.similarity > self.threshold if self.comparator == 'greater' else self.similarity < self.threshold
                self.condition_calls += 1
            if not self.active or self.alpha == 0:
                return output
            value = h+self.alpha*self.vectors[layer]
            old, new = h.norm(dim=-1, keepdim=True), value.norm(dim=-1, keepdim=True)
            if bool((new > old).any()):
                value = value*(old/new.clamp_min(torch.finfo(value.dtype).tiny))
            if not bool(torch.isfinite(value).all()):
                raise FloatingPointError('Nonfinite multi-layer residual; no silent sample exclusion')
            self.edits[layer] += 1
            return (value,)+output[1:] if isinstance(output, tuple) else value
        return apply

    def audit(self):
        return dict(condition_calls=self.condition_calls, condition_active=self.active, condition_score=self.similarity,
            layer_calls=dict(self.calls), layer_residual_edits=dict(self.edits), attention_edits=0,
            manual_refusal_replacements=0, extra_model_calls=0, prefix_replays=0,
            method='CAST-MultiLayer ablation; per-layer paired PCA, not canonical single-layer CAST', alpha=self.alpha)

    def close(self):
        for handle in self.handles:
            handle.remove()


@torch.inference_mode()
def fit_multi(runtime, calibration, first, out, old_weights):
    layers = list(range(first, len(runtime.model.model.layers)+1))
    capture = LayerCapture(runtime.model, layers)
    before = runtime.calls
    pairs = defaultdict(list)
    for row in calibration:
        pairs[row['fact_id']].append(row)
    ragpos, ragneg, refpos, refneg = [], [], [], []
    excluded = []
    def extract(inputs, positions):
        capture.positions = positions; capture.values = {}
        runtime.model(**inputs, use_cache=False, num_logits_to_keep=1)
        return np.stack([capture.values[l] for l in layers])
    try:
        for i, (fid, pair) in enumerate(sorted(pairs.items())):
            wrong = next(r for r in pair if not r['context_correct'])
            if wrong['prior_correct']:
                inputs, _ = runtime.encode(wrong['question'], wrong['context'])
                ragneg.append(extract(inputs, [-1]))
                ragpos.append(extract(*custom_prompt(runtime, wrong['question'])))
            try:
                pos = custom_prompt(runtime, wrong['question'], wrong['context'], suffix='I do not know.')
                neg = custom_prompt(runtime, wrong['question'], wrong['context'], suffix=wrong['designated_answer'])
            except ValueError:
                excluded.append(fid)
            else:
                refpos.append(extract(*pos)); refneg.append(extract(*neg))
            if (i+1) % 50 == 0:
                print(dict(stage='multilayer_calibration_prefills', completed=i+1, total=len(pairs)), flush=True)
    finally:
        capture.close()
    vectors, check = {}, {}
    with threadpool_limits(limits=4):
        for name, positive, negative in [('cast_rag', ragpos, ragneg), ('cast_refusal', refpos, refneg)]:
            positive, negative = np.asarray(positive), np.asarray(negative)
            for j, layer in enumerate(layers):
                vectors[f'{name}_{layer}'] = gram_pc(positive[:, j], negative[:, j])
            cosine = float(vectors[f'{name}_{first}'] @ old_weights[f'{name}_{first}'])
            assert cosine > .99, 'Early-layer source pairing must agree with original PCA direction'
            check[name] = dict(pairs=len(positive), early_direction_cosine_vs_frozen=cosine)
    np.savez_compressed(out / 'multilayer_directions.npz', **vectors)
    dump(out / 'multilayer_fit.json', dict(source_split='calibration', layers=layers, diagnostics=check,
        excluded_suffix_pairs=excluded, offline_forward_calls=runtime.calls-before,
        precision='float64 Gram eigensolve, equivalent leading pair-centered PCA; online fp16',
        unchanged_single_layer_weights=True))
    return vectors


def fluency_proxies(rows):
    # These are operational indicators, not human fluency judgments.
    rates = []
    for row in rows:
        words = row['answer'].lower().split()
        grams = [tuple(words[i:i+3]) for i in range(max(0, len(words)-2))]
        rates.append(1-len(set(grams))/len(grams) if grams else 0.)
    return dict(token_cap12_rate=float(np.mean([r['generated_tokens'] == 12 for r in rows])),
        empty_output_rate=float(np.mean([not r['answer'].strip() for r in rows])),
        mean_repeated_word_trigram_fraction=float(np.mean(rates)),
        spontaneous_refusal_rate=float(np.mean([is_refusal(r['answer']) for r in rows])),
        limitation='automatic truncation/empty/repetition proxies; no human fluency score or proof of KV causal overwrite')


def main(model):
    assert Path('/kaggle/working').is_dir()
    seed_all(42)
    out = ROOT / 'results/cast_sensitivity_v2' / model
    out.mkdir(parents=True, exist_ok=True)
    assert not (out / 'entry_lock.json').exists()
    p = profile(model)
    dump(out / 'entry_lock.json', dict(seed=42, old_blind_quality_access=False, fixed_old_blind_timing_inputs=8, old_blind_tables_modified=False,
        source_profiles_sha256=sha(ROOT / 'configs/nqswap_source_profiles_v2.json')))
    runtime = runtime_for(model, 12)
    dump(out / 'environment.json', environment(runtime, model))
    native = read_jsonl(ROOT / 'data/processed/cast_sensitivity_v2' / model / 'development.jsonl')
    val = [r for r in native if r['split'] == 'val']; cal = [r for r in native if r['split'] == 'calibration']
    old = np.load(ROOT / p['baseline_directions'], allow_pickle=False)
    weights = {k: old[k].copy() for k in old.files}
    multi = fit_multi(runtime, cal, p['layer'], out, weights)
    (out / 'validation').mkdir()
    summaries = []
    for kind in ['cast_rag', 'cast_refusal']:
        for scope, layers in [('single', p['diagnostic_layers']), ('multi', [p['layer']])]:
            for layer in layers:
                cond = p['condition_specs'][str(layer)]
                for alpha in [4., 8., 16.]:
                    name = f'{kind}_{scope}_l{layer}_a{alpha:g}'
                    if scope == 'single':
                        controller = ResidualController(runtime.model, SteeringSpec(kind, layer, alpha,
                            cond['threshold'], cond['comparator']), weights[f'{kind}_{layer}'], weights[f'condition_{layer}'])
                    else:
                        vectors = {l: multi[f'{kind}_{l}'] for l in range(layer, len(runtime.model.model.layers)+1)}
                        controller = MultiCAST(runtime.model, layer, vectors, weights[f'condition_{layer}'],
                                               cond['threshold'], cond['comparator'], alpha)
                    records = []
                    failure = None
                    try:
                        for i, row in enumerate(val):
                            # Do not use generic canonical-controller factory for multi-layer ablation.
                            r = generate(runtime, row['question'], row['context'], controller=controller)
                            item = dict(row, **r, method=name, seed=42)
                            append_jsonl(out / 'validation' / (name+'.jsonl'), item); records.append(item)
                            if (i+1) % 100 == 0:
                                print(dict(stage='cast_strength_validation', variant=name, completed=i+1, total=len(val)), flush=True)
                    except FloatingPointError as exc:
                        failure = dict(variant=name, status='NUMERICAL_FAILURE_INELIGIBLE',
                            completed_validation_rows=len(records), failure=str(exc),
                            policy='retain partial raw outputs, no per-sample deletion; do not select failed variant')
                        append_jsonl(out / 'failed_variants.jsonl', failure)
                    finally:
                        controller.close()
                    if failure is not None:
                        summaries.append(failure)
                        continue
                    summaries.append(dict(variant=name, quality=objective_quality(records, p['source_prior']),
                        fluency_proxies=fluency_proxies(records), controller=scope,
                        spec=dict(method=kind, layer=layer, alpha=alpha, threshold=cond['threshold'],
                                  comparator=cond['comparator'], ooi_normalize=True)))
    dump(out / 'validation_summary.json', summaries)
    canonical = dict(controller='single', spec=p['baseline_choices']['cast']['spec'],
        quality=p['baseline_choices']['cast']['validation'], temperatures=p['baseline_choices']['cast']['temperatures'],
        directions=p['baseline_directions'], condition_directions=p['baseline_directions'], variant='StepB_frozen_cast')
    canonical['quality'] = dict(canonical['quality'])
    rates = canonical['quality']['rates']
    canonical['quality']['prior_utility'] = (1-p['source_prior'])/2*(rates['ext']+rates['abstain'])+p['source_prior']/2*(rates['param']+rates['align'])
    canonical['quality']['mean_utility'] = sum(canonical['quality'][k] for k in ['raw_utility', 'prior_utility', 'macro_utility'])/3
    successful = [r for r in summaries if 'quality' in r]
    best = sorted(successful, key=lambda r: (-r['quality']['macro_utility'], -r['quality']['h_score'],
        r['spec']['alpha'], r['spec']['layer'], r['variant']))[0] if successful else None
    if best is not None and (best['quality']['macro_utility'], best['quality']['h_score']) > (canonical['quality']['macro_utility'], canonical['quality']['h_score']):
        selected = dict(best)
        selected['directions'] = (out / 'multilayer_directions.npz').relative_to(ROOT).as_posix() if best['controller'] == 'multi' else p['baseline_directions']
        selected['condition_directions'] = p['baseline_directions']
        folds = fit_five_temperatures(read_jsonl(out / 'validation' / (best['variant']+'.jsonl')))
        selected.update(temperatures=[f['temperature'] for f in folds], temperature_folds=folds)
    else:
        selected = canonical
    selected.update(source_selection_only=True, nq_inputs_accessed=False, old_blind_tables_modified=False,
        selection_rule='source validation Macro then H; prefer canonical on exact ties; otherwise lower alpha/layer/variant',
        directions_sha256=sha(ROOT / selected['directions']))
    dump(out / 'nq_cast_freeze.json', selected)
    # Canonical frozen baselines, not newly selected high-alpha variants.
    encoder = InteractionScore()
    gate = ActiveFoldedGate(ROOT / p['gate_weights'], p['gate_policy'], 'cuda:0')
    inputs = read_jsonl(ROOT / 'data/processed/cast_sensitivity_v2' / model / 'fixed_inputs.jsonl')
    assert len(inputs) == 8
    methods = ['vanilla', 'cad', 'interaction', 'cast', 'contextfocus', 'active_block']
    if selected['variant'] != 'StepB_frozen_cast':
        methods.append('cast_strong')
    runtime.fixed_length_benchmark = True
    try:
        warm = [next(r for r in inputs if r['state'] == s) for s in ['ext', 'param']]
        for row in warm:
            for method in methods:
                r = request(runtime, row, model, method, encoder=encoder, gate=gate if method == 'active_block' else None, top10=False)
                append_jsonl(out / 'warmup.jsonl', dict(r, warmup=True, fixed_budget=True))
        fixed = []
        for rep in range(3):
            for i, row in enumerate(inputs):
                offset = (rep+i) % len(methods)
                for order, method in enumerate(methods[offset:]+methods[:offset]):
                    r = request(runtime, row, model, method, encoder=encoder,
                                gate=gate if method == 'active_block' else None, top10=False)
                    assert r['generated_tokens'] == 12
                    if method == 'active_block':
                        assert (r['control']['route'] == 1) == (row['expected_original_route'] in [1, 3])
                    r.update(repeat=rep, input_index=i, method_order=order, fixed_budget=True, warmup=False)
                    append_jsonl(out / 'fixed_workload.jsonl', r); fixed.append(r)
        assert len(fixed) == 24*len(methods) and sum(r['method'] == 'active_block' and r['control']['route'] == 1 for r in fixed) == 12
    finally:
        runtime.fixed_length_benchmark = False
    dump(out / 'completion.json', dict(complete=True, model=model, diagnostic_validation_configs=18,
        source_validation_rows=len(val), canonical_timing_requests=144, total_timing_requests=24*len(methods), warmup_requests=2*len(methods),
        actual_active_timing_requests=12, old_blind_quality_model_calls=0, total_model_forward_calls=runtime.calls,
        old_main_baseline_selection_changed=False, nq_cast_strongest_source_frozen=selected['variant'], attention_edits_for_cast=0,
        timing_scope='same-job synchronized request, warmed shared residency; controller registration/static setup excluded for every method; vocabulary logging disabled uniformly'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--model', choices=['qwen3b', 'qwen7b'], required=True)
    main(parser.parse_args().model)
