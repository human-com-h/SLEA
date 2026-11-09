"""Run the locked 646-fact confirmation once, without fitting."""
import argparse
from collections import Counter
import json
import os
from pathlib import Path

import numpy as np

from .active_block_closure import ActiveFoldedGate
from .calibration import utility_correct
from .causalgate_features_v1 import CausalFeatures
from .causalgate_micro_rollback import MicroRollback
from .causalgate_rollback_job import audit, compare, frozen_decision, observed
from .common import ROOT, dump, read_jsonl, seed_all, write_jsonl
from .nqswap_job_v2 import closed_book
from .nqswap_runtime_v2 import environment, metadata, native_rows, request, runtime_for, sha
from .output_gates import InteractionScore
from .phase4_common import append_jsonl, objective_quality
from .phase4_runtime import ScopeController


def run(model):
    assert Path('/kaggle/working').is_dir()
    os.environ['SLEA_NQ_STRONG_CAST'] = '1'
    config_path = ROOT/'configs/nqswap_stage_b_freeze.json'
    config = json.loads(config_path.read_text(encoding='utf-8'))
    settings = config['models'][model]
    inputs = ROOT/'data/processed/nqswap_stage_b_646'
    lockpath = inputs/'sample_lock.json'
    lock = json.loads(lockpath.read_text(encoding='utf-8'))
    assert lock['freeze_sha256']==sha(config_path)
    assert config['source_profiles_sha256']==sha(ROOT/'configs/nqswap_source_profiles_v2.json')
    for name, asset in settings['assets'].items():
        assert sha(ROOT/name)==asset['sha256']
    assert sha(inputs/'facts.jsonl')==lock['files']['facts.jsonl']
    facts = read_jsonl(inputs/'facts.jsonl')
    assert len(facts)==646 and [f['fact_id'] for f in facts]==lock['fact_ids']
    assert len(set(lock['component_ids']))==646 and [f['component_id'] for f in facts]==lock['component_ids']
    out = ROOT/'results/nqswap_stage_b'/model
    out.mkdir(parents=True)
    seed_state = seed_all(config['seed'])
    dump(out/'entry.json',dict(freeze_sha256=sha(config_path),sample_lock_sha256=sha(lockpath),seed=seed_state))
    runtime = runtime_for(model,config['quality']['max_new_tokens'])
    env = environment(runtime,model)
    assert env['all_parameters_frozen_fp16']
    dump(out/'environment.json',env)
    labels = closed_book(runtime,facts,out/'closed_book.jsonl')
    rows = native_rows(facts,labels)
    assert len(rows)==1292 and len({r['id'] for r in rows})==1292
    write_jsonl(out/'labels.jsonl',[metadata(r) for r in rows])
    prior = sum(r['prior_correct'] for r in labels.values())/len(facts)
    dump(out/'label_lock.json',dict(sha256=sha(out/'labels.jsonl'),
        closed_book_sha256=sha(out/'closed_book.jsonl'),p_cb=prior,counts=Counter(r['state'] for r in rows)))

    spec = settings['gate']
    with np.load(ROOT/settings['weights'],allow_pickle=False) as saved:
        weights = {key:saved[key] for key in saved.files if not key.startswith('val_')}
    layers = {l for role in ('known','conflict') for f in spec[role] for l in f['layers']}
    collector = CausalFeatures(runtime.model,layers,group='last')
    score = {}
    def record(value):
        score.update(value)
    gate = MicroRollback(runtime,spec['selected']['control'],spec['selected']['read'],collector,
                         frozen_decision(spec,weights,record=record),first_masked=settings['first_masked'])
    hard = {l:ScopeController(runtime,l,force_route=1) for l in settings['oracle_layers']}
    original_spec = settings['original']
    original = ScopeController(runtime,original_spec['layer'],gate=ActiveFoldedGate(
        ROOT/original_spec['weights'],original_spec['policy'],'cuda:0'))
    encoder = InteractionScore()
    branches = {}
    def save(name,result):
        record = dict(metadata(row),**result)
        append_jsonl(out/f'{name}.jsonl',record)
        branches.setdefault(name,[]).append(record)

    discrepancies = Counter()
    for i,row in enumerate(rows,1):
        vanilla = observed(runtime,row)
        save('vanilla',vanilla)
        score.clear()
        online = observed(runtime,row,gate)
        control = audit(gate,online,online['prompt_tokens'])
        assert set(score)=={'known_z','conflict_z','joint'}
        assert gate.attention_calls == (
            (env['layers'] - settings['first_masked'] + 1) * online['generated_tokens'] * control['blocked'])
        save('primary',dict(online,gate_scores=dict(score),control=control))
        references = {}
        for layer,controller in hard.items():
            references[layer] = observed(runtime,row,controller)
            save(f'hard_l{layer}',references[layer])
        selected_hard = references[spec['selected']['control']]
        expected = selected_hard if control['blocked'] else vanilla
        agreement = compare(online,expected)
        for field,value in agreement.items():
            discrepancies[field] += int(value is not True)
        balanced_block = score['known_z']>=0 and score['joint']>=settings['balanced']['threshold']
        balanced_reference = selected_hard if balanced_block else vanilla
        append_jsonl(out/'audit.jsonl',dict(id=row['id'],primary_block=control['blocked'],
            balanced_block=balanced_block,matched_reference=agreement))
        def reuse(result, blocked):
            return {**{k:result[k] for k in ('answer','generated_token_ids','first_logits_sha256','first_top10')},
                    'blocked':bool(blocked),'cache_composed':True}
        save('balanced',reuse(balanced_reference,balanced_block))
        for layer,branch in references.items():
            wrong_context = not row['context_correct']
            save(f'oracle_l{layer}',reuse(branch if wrong_context else vanilla,wrong_context))
        result = observed(runtime,row,original)
        original_audit = original.audit()
        assert original_audit['gate_calls']==1 and result['forward_calls']==result['generated_tokens']
        probabilities = original_audit['route_probabilities']
        save('original',dict(result,control=original_audit,conflict_score=probabilities[1]+probabilities[3]))
        # Scope controllers persist across requests. Reset them even on the 3B runtime.
        for controller in runtime.model._phase4_controllers:
            controller.reset(None)
        for method in ('cast_strong','contextfocus','interaction'):
            result = request(runtime,row,model,method,encoder)
            append_jsonl(out/f'{method}.jsonl',result)
            branches.setdefault(method,[]).append(result)
        if i%25==0 or i==1292:
            print(dict(model=model,completed=i,total=1292,primary_audit_mismatches=dict(discrepancies)),flush=True)
    for controller in [gate,original,*hard.values()]:
        controller.close()
    collector.close()
    # Serialize a per-row utility check for traceability, alongside the original responses.
    summary = {name:objective_quality(records,prior) for name,records in branches.items()}
    for name,records in branches.items():
        assert len(records)==1292
        summary[name]['correct_count']=sum(utility_correct(r) for r in records)
    dump(out/'completion.json',dict(complete=True,model=model,contexts=len(rows),offline_closed_book_requests=len(facts),
        quality_requests=len(rows)*(6+len(hard)),total_model_forward_calls=runtime.calls,
        balanced_additional_model_requests=0,primary_equivalence_mismatches=discrepancies,points=summary))


if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model',choices=['qwen3b','qwen7b'],required=True)
    run(parser.parse_args().model)
