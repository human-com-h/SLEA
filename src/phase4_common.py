"""Shared new-model protocol checks, requests and validation-only calibration."""
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit, logit
from sklearn.model_selection import GroupKFold
from .common import ROOT, STATES, exact_match, is_refusal, dump
from .calibration import utility_correct, raw_confidence, fit_temperature
from .phase4_runtime import ScopeController, synchronize


def load_protocol(stage):
    cfgpath = ROOT/'configs/phase4_qwen7b_v1.json'
    cfg = json.loads(cfgpath.read_text())
    lock = json.loads((ROOT/'data/processed/phase4_qwen7b_v1/input_lock.json').read_text())
    assert hashlib.sha256(cfgpath.read_bytes()).hexdigest() == lock['protocol_sha256']
    assert hashlib.sha256((ROOT/'docs/phase4_preregistration.md').read_bytes()).hexdigest() == lock['preregistration_sha256']
    name = 'blind_facts.jsonl' if stage == 'blind' else 'development_facts.jsonl'
    path = ROOT/'data/processed/phase4_qwen7b_v1'/name
    assert hashlib.sha256(path.read_bytes()).hexdigest() == lock['files'][name]
    other = 'development_facts.jsonl' if stage == 'blind' else 'blind_facts.jsonl'
    assert not (path.parent/other).exists(), 'Physical development/blind separation required in cloud package'
    return cfg, lock, path


def native_rows(facts, closed):
    result = []
    for f in facts:
        prior = closed[f['fact_id']]
        for c in f['contexts']:
            state = ('align' if c['context_correct'] else 'param') if prior['prior_correct'] else ('ext' if c['context_correct'] else 'abstain')
            result.append({k: f[k] for k in ['fact_id', 'question', 'gold_answers', 'subject_id', 'fact_triple', 'component_id', 'split', 'split_id']} |
                          {**c, 'id': f['fact_id']+'_'+c['variant'], 'prior_correct': prior['prior_correct'],
                           'prior_answer': prior['answer'], 'state': state})
    return result


def row_metadata(row):
    return {k: row[k] for k in ['id', 'fact_id', 'state', 'gold_answers', 'split', 'split_id', 'prior_correct', 'component_id']}


def controller_for(runtime, method, cfg, layer, gate=None):
    if method in ['early_gate', 'last_query']:
        return ScopeController(runtime, layer, gate=gate, scope='last_query' if method == 'last_query' else 'all_queries')
    if method in ['oracle', 'static_hard']: return ScopeController(runtime, layer, force_route=1)
    if method == 'static_span': return ScopeController(runtime, cfg['static_span_layer'], force_route=1, scope='last_query', soft_bias=cfg['static_span_bias'])
    return None


def request(runtime, row, method, cfg, controller=None, encoder=None, threshold=None):
    for i in range(2): torch.cuda.reset_peak_memory_stats(i)
    if method == 'interaction':
        import time
        synchronize(); start = time.perf_counter()
        cb = runtime.generate(row['question'])
        score, encoder_ms = encoder.score(cb['answer'], row['context'], cb['first_token']['confidence'])
        selected_cb = score > threshold
        result = dict(cb) if selected_cb else runtime.generate(row['question'], row['context'])
        synchronize(); total = (time.perf_counter()-start)*1000
        result.update(latency_ms=total, ttft_ms=total if selected_cb else cb['latency_ms']+encoder_ms+result['ttft_ms'],
                      source_selected='closed_book' if selected_cb else 'rag', interaction_score=score,
                      sentence_encoder_ms=encoder_ms, closed_book_ms=cb['latency_ms'],
                      closed_book_generated_tokens=cb['generated_tokens'],
                      total_model_generated_tokens=cb['generated_tokens']+(0 if selected_cb else result['generated_tokens']))
        if not selected_cb: result['forward_calls'] += cb['forward_calls']
        assert result['forward_calls'] == result['total_model_generated_tokens']
    else:
        if method == 'oracle': controller.force_route = STATES.index(row['state'])
        result = runtime.generate(row['question'], row['context'], controller=controller,
                                  cad_alpha=cfg['cad_alpha'] if method == 'cad' else None)
        assert result['forward_calls'] == result['generated_tokens']*(2 if method == 'cad' else 1)
        if controller:
            control = controller.audit(); result['control'] = control
            assert control['gate_calls'] == 1 and control['middle_calls'] == result['generated_tokens']
            assert control['query_lengths'] == [result['prompt_tokens']]+[1]*(result['generated_tokens']-1)
            assert control['residual_edits'] == 0 and control['extra_model_calls'] == 0
            assert control['attention_calls'] == (len(runtime.model.model.layers)-control['gate_layer'])*result['generated_tokens']*control['block']
            if method in ['early_gate', 'last_query', 'oracle'] and control['route'] == 3:
                result.update(raw_generated_answer=result['answer'], answer='I do not know', refusal_policy='post_generation_replace')
    return row_metadata(row) | result | {'method': method, 'seed': 42}


def objective_quality(rows, prior):
    rates = {s: float(np.mean([utility_correct(r) for r in rows if r['state'] == s])) for s in STATES}
    assert all(np.isfinite(list(rates.values())))
    raw = float(np.mean([utility_correct(r) for r in rows]))
    aligned = (1-prior)/2*(rates['ext']+rates['abstain'])+prior/2*(rates['param']+rates['align'])
    macro = sum(rates.values())/4
    h = 2*rates['ext']*rates['param']/(rates['ext']+rates['param']) if rates['ext']+rates['param'] else 0.
    refusal = np.array([is_refusal(r['answer']) for r in rows]); abstain = np.array([r['state'] == 'abstain' for r in rows])
    f1 = 2*float(np.sum(refusal & abstain))/(refusal.sum()+abstain.sum()) if refusal.sum()+abstain.sum() else 0.
    return {'rates': rates, 'raw_utility': raw, 'prior_utility': aligned, 'macro_utility': macro,
            'mean_utility': (raw+aligned+macro)/3, 'h_score': h, 'refusal_f1': float(f1)}


def fit_five_temperatures(rows):
    assert all(r['split'] == 'val' for r in rows)
    folds = []
    for train, test in GroupKFold(n_splits=5).split(rows, groups=[r['fact_id'] for r in rows]):
        fit = fit_temperature([rows[i] for i in train])
        folds.append({'temperature': fit['temperature'], 'training_facts': len({rows[i]['fact_id'] for i in train}),
                      'held_out_fact_ids': sorted({rows[i]['fact_id'] for i in test})})
    return folds


def attach_probability(row, temperatures):
    raw = float(np.clip(raw_confidence(row), 1e-6, 1-1e-6))
    return dict(row, utility_confidence_raw=raw,
                utility_confidence_frozen=float(np.mean([expit(logit(raw)/t) for t in temperatures])),
                confidence_temperatures=temperatures)


def append_jsonl(path, item):
    with Path(path).open('a', encoding='utf-8') as f: f.write(json.dumps(item, ensure_ascii=False, allow_nan=False)+'\n')
