"""Technical two-GPU smoke; no blind inputs, no parameter selection."""
import json
import math
import numpy as np
import torch
from .common import ROOT, dump, read_jsonl, seed_all
from .phase4_runtime import Phase4Runtime, ScopeController, Phase4Gate


def smoke(runtime, cfg, directory):
    directory.mkdir(parents=True, exist_ok=True)
    fact = read_jsonl(ROOT/'data/processed/phase4_qwen7b_v1/development_facts.jsonl')[0]
    assert fact['split'] == 'calibration'
    context = fact['contexts'][0]['context']; question = fact['question']; layer = cfg['layers'][0]
    runtime.capture_first_logits = True
    runtime.generate(question, context)
    baseline = runtime.generate(question, context); base_logits = runtime.last_first_logits.clone()
    repeat = runtime.generate(question, context); repeat_logits = runtime.last_first_logits.clone()
    controller = ScopeController(runtime, layer, force_route=0)
    try:
        hold = runtime.generate(question, context, controller=controller); hold_logits = runtime.last_first_logits.clone()
        assert controller.audit()['attention_calls'] == 0
    finally: controller.close()
    def logit_comparison(a, b):
        finite = torch.isfinite(a) & torch.isfinite(b)
        return {'exact': torch.equal(a, b), 'a_nonfinite': int((~torch.isfinite(a)).sum()),
                'b_nonfinite': int((~torch.isfinite(b)).sum()),
                'max_abs_finite_difference': float((a[finite]-b[finite]).abs().max()) if finite.any() else None}
    diagnostics = {'baseline': baseline, 'repeat': repeat, 'hold': hold,
                   'repeat_logits': logit_comparison(base_logits, repeat_logits),
                   'hold_logits': logit_comparison(base_logits, hold_logits)}
    def safe(value):
        if isinstance(value, float) and not math.isfinite(value): return None
        if isinstance(value, dict): return {k: safe(v) for k, v in value.items()}
        if isinstance(value, list): return [safe(v) for v in value]
        return value
    dump(directory/'zero_intervention_diagnostics.json', safe(diagnostics))
    print({'zero_intervention_diagnostics': safe(diagnostics)}, flush=True)
    assert torch.equal(base_logits, hold_logits) and baseline['generated_token_ids'] == hold['generated_token_ids']
    active = {}
    for scope in ['all_queries', 'last_query']:
        controller = ScopeController(runtime, layer, scope=scope, force_route=1)
        try:
            row = runtime.generate(question, context, controller=controller)
            audit = controller.audit()
            assert audit['attention_calls'] == (len(runtime.model.model.layers)-layer)*row['generated_tokens']
            assert audit['gate_calls'] == 1 and audit['middle_calls'] == row['generated_tokens']
            assert row['forward_calls'] == row['generated_tokens']
            diff = float((base_logits-runtime.last_first_logits).abs().max())
            assert diff > 0
            active[scope] = {'generation': row, 'audit': audit, 'first_logits_max_abs_change': diff}
        finally: controller.close()
    d = runtime.model.config.hidden_size
    path = directory/'constant_gate.npz'
    np.savez(path, weights=np.zeros((1, d, 4), dtype=np.float16), bias=np.array([-10, 10, -10, -10], dtype=np.float16), layers=[layer])
    policy = {'param_cost': 1., 'abstain_cost': 1., 'param_threshold': .5, 'abstain_threshold': .5}
    gate = Phase4Gate(path, policy, next(runtime.model.model.layers[layer-1].parameters()).device)
    controller = ScopeController(runtime, layer, gate=gate)
    try:
        row = runtime.generate(question, context, controller=controller)
        assert row['generated_token_ids'] == active['all_queries']['generation']['generated_token_ids']
        assert controller.audit()['gate_calls'] == 1
    finally: controller.close()
    assert set(baseline['cache_devices']) == {'cuda:0', 'cuda:1'}
    runtime.fixed_length_benchmark = True
    fixed = runtime.generate(question, context, cad_alpha=cfg['cad_alpha'])
    runtime.fixed_length_benchmark = False; runtime.capture_first_logits = False
    assert fixed['forward_calls'] == 2*cfg['max_new_tokens'] and fixed['generated_tokens'] == cfg['max_new_tokens']
    result = {'all_checks_passed': True, 'technical_only': True, 'blind_model_calls': 0,
              'zero_intervention_logits_and_tokens_exact': True, 'active_masks_on_both_devices': active,
              'learned_gate_interface_single_prefill_verified': True, 'cache_devices': baseline['cache_devices'],
              'fixed_cad_calls': fixed['forward_calls'], 'fact_id': fact['fact_id'], 'layer': layer}
    dump(directory/'smoke_verification.json', result)
    print({'smoke_complete': True, 'layer': layer, 'blind_model_calls': 0}, flush=True)
    return result


if __name__ == '__main__':
    cfg = json.loads((ROOT/'configs/phase4_qwen7b_v1.json').read_text())
    seed_all(42); runtime = Phase4Runtime(cfg)
    from .phase4_numerics import disable_reduced_accumulation, prescale_query, finite_observers
    directory = ROOT/'results/phase4_qwen7b_v1/smoke'
    disable_reduced_accumulation()
    observations, handles = finite_observers(runtime.model)
    try:
        try:
            smoke(runtime, cfg, directory/'no_reduced_accumulation')
            choice = {'disable_reduced_accumulation': True, 'query_prescale': False}
        except AssertionError:
            for h in handles: h.remove()
            handles = []
            dump(directory/'fp16_finiteness_diagnostics.json', observations)
            change = prescale_query(runtime.model)
            smoke(runtime, cfg, directory/'prescaled_query')
            choice = {'disable_reduced_accumulation': True, 'query_prescale': True, 'implementation': change}
        dump(directory/'numerical_choice.json', choice)
        print({'numerical_choice': choice}, flush=True)
    finally:
        for h in handles: h.remove()
