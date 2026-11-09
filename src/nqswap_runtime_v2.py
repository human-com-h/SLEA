"""New-study helpers only; original model/controller files remain immutable."""
import hashlib
import json
import os
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT, exact_match, is_refusal
from .model_runtime import Runtime
from .phase4_runtime import Phase4Runtime, ScopeController, FeatureCapture, synchronize
from .active_block_closure import ActiveFoldedGate
from .internal_control_baselines_v1 import ResidualController, SteeringSpec


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def profile(model):
    p = json.loads((ROOT / 'configs/nqswap_source_profiles_v2.json').read_text())[model]
    if os.environ.get('SLEA_NQ_STRONG_CAST') == '1':
        chosen = json.loads((ROOT / 'results/cast_sensitivity_v2' / model / 'nq_cast_freeze.json').read_text())
        p['cast_override'] = chosen
        p['baseline_temperatures']['cast'] = chosen['temperatures']
    return p


def runtime_for(model, max_new=32):
    if model == 'qwen3b':
        runtime = Runtime(max_input=512, max_new=max_new)
        runtime.model._phase4_controllers = []
    else:
        cfg = json.loads((ROOT / 'configs/phase4_qwen7b_v1.json').read_text())
        cfg['max_new_tokens'] = max_new
        runtime = Phase4Runtime(cfg)
    return runtime


def environment(runtime, model):
    import transformers
    import sklearn
    return dict(model=model, torch=torch.__version__, numpy=np.__version__, sklearn=sklearn.__version__,
        transformers=transformers.__version__, backend='eager', precision='float16', seed=42,
        layers=len(runtime.model.model.layers), hidden_size=runtime.model.config.hidden_size,
        gpu_names=[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        device_map=getattr(runtime.model, 'hf_device_map', {}), max_new_tokens=runtime.max_new,
        all_parameters_frozen_fp16=all(not p.requires_grad and p.dtype == torch.float16 for p in runtime.model.parameters()))


class Top10Capture:
    """Observe ordinary model outputs; include the CAD effective first logits."""
    def __init__(self, model, cad_alpha=None):
        self.raw = []
        self.limit = 2 if cad_alpha is not None else 1
        self.alpha = cad_alpha
        self.handle = model.register_forward_hook(self.hook)

    def hook(self, module, args, output):
        if len(self.raw) < self.limit:
            self.raw.append(output.logits[0, -1].detach().clone())

    @staticmethod
    def top(logits):
        values, ids = logits.topk(10)
        return dict(token_ids=ids.cpu().tolist(), logits=values.float().cpu().tolist(), dtype=str(logits.dtype))

    def finish(self):
        assert len(self.raw) == self.limit
        result = dict(first_forward=self.top(self.raw[0]))
        if self.alpha is not None:
            result['closed_book_reference_forward'] = self.top(self.raw[1])
            result['effective_generation'] = self.top((1+self.alpha)*self.raw[0]-self.alpha*self.raw[1])
        else:
            result['effective_generation'] = result['first_forward']
        self.handle.remove()
        return result

    def close(self):
        self.handle.remove()


def native_rows(facts, closed):
    result = []
    for f in facts:
        cb = closed[f['fact_id']]
        known = exact_match(cb['answer'], f['org_answer'])
        for positive, side in [(True, 'org_context'), (False, 'sub_context')]:
            state = ('align' if positive else 'param') if known else ('ext' if positive else 'abstain')
            result.append(dict(id=f['fact_id']+('_org' if positive else '_sub'), fact_id=f['fact_id'],
                component_id=f['component_id'], question=f['question'], context=f[side], context_correct=positive,
                gold_answers=f['org_answer'], swapped_answers=f['sub_answer'], state=state, prior_correct=known,
                split=f['split'], split_id=f['split_id']))
    return result


def metadata(r):
    return {k: r[k] for k in ['id', 'fact_id', 'component_id', 'state', 'gold_answers',
        'swapped_answers', 'prior_correct', 'split', 'split_id'] if k in r}


def control_for(runtime, model, method, gate=None):
    p = profile(model)
    if method in ['all_hard', 'last_hard', 'active_block', 'last_active_block']:
        return ScopeController(runtime, p['layer'], scope='last_query' if method in ['last_hard', 'last_active_block'] else 'all_queries',
            gate=gate, force_route=1 if gate is None else None)
    if method == 'cast_strong' or (method == 'cast' and 'cast_override' in p):
        chosen = p.get('cast_override') if method == 'cast' else json.loads((ROOT / 'results/cast_sensitivity_v2' / model / 'nq_cast_freeze.json').read_text())
        spec = chosen['spec']
        with np.load(ROOT / chosen['condition_directions'], allow_pickle=False) as z:
            condition = z[f"condition_{spec['layer']}"].copy()
        with np.load(ROOT / chosen['directions'], allow_pickle=False) as z:
            if chosen['controller'] == 'multi':
                from .cast_sensitivity_v2 import MultiCAST
                vectors = {l: z[f"{spec['method']}_{l}"].copy() for l in range(spec['layer'], len(runtime.model.model.layers)+1)}
                return MultiCAST(runtime.model, spec['layer'], vectors, condition, spec['threshold'], spec['comparator'], spec['alpha'])
            behavior = z[f"{spec['method']}_{spec['layer']}"].copy()
        return ResidualController(runtime.model, SteeringSpec(**spec), behavior, condition)
    if method in ['cast', 'contextfocus']:
        spec = SteeringSpec(**p['baseline_choices'][method]['spec'])
        with np.load(ROOT / p['baseline_directions'], allow_pickle=False) as z:
            behavior = z[f'{spec.method}_{spec.layer}'].copy()
            condition = z[f'condition_{spec.layer}'].copy() if method == 'cast' else None
        return ResidualController(runtime.model, spec, behavior, condition)
    return None


def generate(runtime, question, context=None, controller=None, cad_alpha=None, capture=None, top10=True):
    observer = Top10Capture(runtime.model, cad_alpha) if top10 else None
    try:
        result = runtime.generate(question, context, controller=controller, cad_alpha=cad_alpha, capture=capture)
        if not np.isfinite([result['answer_confidence'], result['first_token']['confidence'],
                            result['first_token']['entropy'], result['first_token']['margin']]).all():
            raise FloatingPointError('Nonfinite generation confidence; no silent repair or sample deletion')
        if observer:
            result['first_token_top10'] = observer.finish()
        if controller is not None:
            audit = controller.audit()
            result['control'] = audit
            assert result['forward_calls'] == result['generated_tokens']
            if 'gate_calls' in audit:
                assert audit['gate_calls'] == 1
                assert audit['middle_calls'] == result['generated_tokens']
                assert audit['attention_calls'] == (len(runtime.model.model.layers)-audit['gate_layer'])*result['generated_tokens']*audit['block']
                assert audit['residual_edits'] == 0
            else:
                assert audit['attention_edits'] == 0 and audit['manual_refusal_replacements'] == 0
    finally:
        if observer:
            observer.close()
    return result


def request(runtime, row, model, method, encoder=None, gate=None, top10=True, capture=None):
    p = profile(model)
    controller = control_for(runtime, model, method, gate)
    synchronize()
    idle = [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
    for i in range(torch.cuda.device_count()):
        torch.cuda.reset_peak_memory_stats(i)
    start = time.perf_counter()
    count_start = runtime.calls
    try:
        if method == 'interaction':
            assert encoder is not None
            cb = generate(runtime, row['question'], top10=top10)
            computed_first = cb['ttft_ms']
            score, encoder_ms = encoder.score(cb['answer'], row['context'], cb['first_token']['confidence'])
            choose_cb = score > p['interaction_threshold']
            result = dict(cb) if choose_cb else generate(runtime, row['question'], row['context'], top10=top10)
            result.update(source_selected='closed_book' if choose_cb else 'rag', interaction_score=score,
                closed_book_generated_tokens=cb['generated_tokens'], sentence_encoder_ms=encoder_ms,
                closed_book_ms=cb['latency_ms'])
            result['ttft_ms'] = ((time.perf_counter()-start)*1000 if choose_cb else cb['latency_ms']+encoder_ms+result['ttft_ms'])
            if top10:
                result['interaction_closed_book_top10'] = cb['first_token_top10']
        else:
            result = generate(runtime, row['question'], row['context'], controller=controller,
                cad_alpha=p['cad_alpha'] if method == 'cad' else None, capture=capture, top10=top10)
            computed_first = result['ttft_ms']
        synchronize()
        elapsed = (time.perf_counter()-start)*1000
        result.update(latency_ms=elapsed, forward_calls=runtime.calls-count_start,
            ttft_first_computed_ms=computed_first, ttft_returnable_ms=result['ttft_ms'],
            idle_allocated_by_gpu=idle,
            peak_allocated_bytes_by_gpu=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
        assert result['forward_calls'] == (result['generated_tokens']*(2 if method == 'cad' else 1) if method != 'interaction'
            else result['closed_book_generated_tokens']+(0 if result['source_selected'] == 'closed_book' else result['generated_tokens']))
        result.update(method=method, seed=42, manual_refusal_replacements=0, extra_arbitration_model_calls=0, prefix_replays=0)
        return dict(metadata(row), **result)
    finally:
        if controller is not None:
            controller.close()


def replay(base, hard, intercepted, method):
    assert len(base) == len(hard) == len(intercepted)
    result = []
    for b, h, block in zip(base, hard, intercepted):
        assert b['id'] == h['id']
        r = dict(h if block else b)
        # Branch timings/calls are not an automatic policy measurement.
        r['source_branch_measurement'] = {k: r.pop(k) for k in ['latency_ms', 'ttft_ms', 'forward_calls'] if k in r}
        r.update(method=method, intercepted=bool(block), cache_composed_quality=True)
        result.append(r)
    return result
