"""GPU engineering audit restricted to the existing validation split."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .causalgate_features_v1 import CausalFeatures
from .causalgate_micro_rollback import MicroRollback
from .common import ROOT, STATES, dump, read_jsonl, write_jsonl, seed_all
from .nqswap_runtime_v2 import runtime_for, request
from .phase4_runtime import ScopeController
from .active_block_closure import InteractionScore
from .calibration import utility_correct


def frozen_decision(spec, weights, record=None):
    selected = spec['selected']
    parameters = {}
    def factor(collector, role, specs):
        scores = []
        for i, features in enumerate(specs):
            vectors = [collector.vector(layer, group) for layer in features['layers']
                       for group in features['group'].split('+')]
            device = vectors[-1].device
            vectors = [v.to(device) for v in vectors]
            x = torch.stack(vectors).mean(0) if features['combine'] == 'mean' else torch.cat(vectors)
            prefix = f'{role}_{i}' if f'{role}_{i}_weight' in weights else role
            key = (prefix, device)
            if key not in parameters:
                parameters[key] = tuple(torch.as_tensor(weights[f'{prefix}_{k}'],device=device)
                                        for k in ['weight','bias'])
            w,b = parameters[key]
            # Match the development CPU scoring contract.
            value = float((x.float() @ w.float() + b.float()).half().item())
            if role == 'known' and spec.get('ensemble'):
                value = (value - float(weights[f'{prefix}_mean'])) / float(weights[f'{prefix}_std'])
            scores.append(value)
        return float(np.mean(scores))
    refs = {role: weights[f'{role}_oof_reference'] for role in ['known','conflict']}
    means = {role:r.mean() for role,r in refs.items()}
    stds = {role:r.std() for role,r in refs.items()}
    refs = {role:np.sort(r) for role,r in refs.items()}
    def decide(collector):
        if selected['threshold'] is None:
            return False
        k = factor(collector,'known',spec['known'])
        c = factor(collector,'conflict',spec['conflict'])
        if selected['gate'] == 'min':
            value = min(np.searchsorted(refs['known'],k,side='right')/len(refs['known']),
                        np.searchsorted(refs['conflict'],c,side='right')/len(refs['conflict']))
            return value >= selected['threshold']
        kz = (k-means['known'])/stds['known']
        cz = (c-means['conflict'])/stds['conflict']
        joint = kz + selected['weight'] * cz
        if record is not None:
            record(dict(known_z=float(kz), conflict_z=float(cz), joint=float(joint)))
        return kz >= 0 and joint >= selected['threshold']
    return decide


def observed(runtime,row,controller=None):
    for installed in runtime.model._phase4_controllers:
        installed.reset(None)
    tokens=[]
    first=None
    def observe(module,args,output):
        nonlocal first
        logits=output.logits[0,-1]
        if first is None:
            first=logits.detach().clone()
        tokens.append(logits.argmax().detach())
    hook=runtime.model.register_forward_hook(observe)
    try:
        result=runtime.generate(row['question'],row['context'],controller=controller)
    finally:
        hook.remove()
    result['generated_token_ids']=torch.stack(tokens).cpu().tolist()
    first=first.cpu().contiguous()
    result['first_logits_sha256']=hashlib.sha256(first.numpy().tobytes()).hexdigest()
    result['first_top10']=dict(zip(['logits','token_ids'],[x.tolist() for x in first.topk(10)]))
    return result


def compare(actual,reference):
    comparisons={}
    for key in ['answer','generated_token_ids','first_logits_sha256']:
        comparisons[key]=None if key not in reference else actual[key]==reference[key]
    cached=reference.get('first_token_top10',{}).get('effective_generation')
    if cached:
        comparisons['first_top10']=all(actual['first_top10'][k]==cached[k] for k in ['logits','token_ids'])
    return comparisons


def audit(controller,result,prompt):
    n=result['generated_tokens']
    count=[n]*len(controller.layers)
    if controller.block:
        for i in range(controller.first_masked-1,controller.read):
            count[i]+=1
    assert controller.calls_by_layer==count and result['forward_calls']==n
    assert controller.gate_calls==1
    assert controller.cache_steps==[[prompt+step]*len(count) for step in range(n)]
    return dict(layer_calls=count,cache_lengths=controller.cache_steps,
                discarded_layers=controller.discarded_layers,blocked=controller.block,
                replay_gpu_ms=sum(a.elapsed_time(b) for a,b in controller.replay_events))


def main(model):
    seed_all(42)
    p=ROOT/'data/processed/causalgate_stage_a_v1'/model
    bundle=ROOT/'configs/rollback_development'
    spec=json.loads((bundle/'candidate.json').read_text())
    with np.load(bundle/'candidate.npz') as z:weights={k:z[k] for k in z.files}
    rows=[r for r in read_jsonl(p/'development.jsonl') if r['split']=='val']
    assert len(rows)==300
    base={r['id']:r for r in read_jsonl(p/'validation_vanilla.jsonl')}
    ctrl=spec['selected']['control']
    read=spec['selected']['read']
    hard={r['id']:r for r in read_jsonl(bundle/'validation_hard.jsonl')}
    expected=np.asarray(spec['expected_routes'],bool)
    assert [r['id'] for r in rows]==spec['ids']
    runtime=runtime_for(model,32)
    layers={l for role in ['known','conflict'] for f in spec[role] for l in f['layers']}
    collector=CausalFeatures(runtime.model,sorted(layers),group='all_causal' if any(
        'lens' in f['group'] for role in ['known','conflict'] for f in spec[role]) else 'last')
    decide=frozen_decision(spec,weights)
    rollback=MicroRollback(runtime,ctrl,read,collector,decide)
    reference=ScopeController(runtime,ctrl,force_route=1)
    out=ROOT/'results/causalgate_rollback_v2_1'/model
    out.mkdir(parents=True,exist_ok=True)
    results=[]
    try:
        for idx,row in enumerate(rows):
            actual=observed(runtime,row,rollback)
            control=audit(rollback,actual,actual['prompt_tokens'])
            expected_route=bool(expected[idx])
            cached=(hard if expected_route else base)[row['id']]
            fresh=observed(runtime,row,reference if expected_route else None)
            results.append(dict(id=row['id'],state=row['state'],expected_route=expected_route,
                control=control,online=actual,fresh=compare(actual,fresh),cached=compare(actual,cached)))
            if (idx+1)%25==0:print(model,'validation',idx+1,flush=True)
        write_jsonl(out/'validation.jsonl',results)
        timing_rows=[]
        for state in STATES:
            subset=[i for i,r in enumerate(rows) if r['state']==state]
            subset.sort(key=lambda i:not expected[i])
            timing_rows.extend(rows[i] for i in subset[:2])
        input_boundary = MicroRollback(runtime,ctrl,read,collector,decide,first_masked=ctrl)
        input_reference = ScopeController(runtime,ctrl-1,force_route=1)
        input_checks = []
        try:
            for row in timing_rows:
                actual = observed(runtime,row,input_boundary)
                control = audit(input_boundary,actual,actual['prompt_tokens'])
                fresh = observed(runtime,row,input_reference if control['blocked'] else None)
                input_checks.append(dict(id=row['id'],control=control,fresh=compare(actual,fresh)))
        finally:
            input_boundary.close()
            input_reference.close()
        write_jsonl(out/'input_boundary_checks.jsonl',input_checks)
        runtime.max_new=12
        runtime.fixed_length_benchmark=True
        encoder=InteractionScore()
        def timed(row,method):
            for installed in runtime.model._phase4_controllers:
                installed.reset(None)
            if method=='rollback':
                result=runtime.generate(row['question'],row['context'],controller=rollback)
                control=audit(rollback,result,result['prompt_tokens'])
                return dict(result,replay_gpu_ms=control['replay_gpu_ms'],blocked=control['blocked'])
            if method=='vanilla':
                return runtime.generate(row['question'],row['context'])
            return request(runtime,row,model,method,encoder=encoder,top10=False)
        methods=['vanilla','rollback','cad','interaction']
        for method in methods:timed(timing_rows[0],method)
        timing=[]
        for repeat in range(3):
            for i,row in enumerate(timing_rows):
                shift=(repeat+i)%len(methods)
                for method in methods[shift:]+methods[:shift]:
                    timing.append(dict(id=row['id'],method=method,repeat=repeat,**{
                        key:value for key,value in timed(row,method).items()
                        if key in ['ttft_ms','latency_ms','replay_gpu_ms','blocked','forward_calls','generated_tokens']}))
        write_jsonl(out/'timing.jsonl',timing)
        averages={m:{k:dict(mean=float(np.mean(v)),sd=float(np.std(v,ddof=1)))
                     for k in ['ttft_ms','latency_ms','replay_gpu_ms']
                     if (v:=[r[k] for r in timing if r['method']==m and k in r])}
                  for m in methods}
        points={s:float(np.mean([utility_correct(dict(row,**r['online'])) for row,r in zip(rows,results)
                                if row['state']==s])) for s in STATES}
        dump(out/'summary.json',dict(timing=averages,online_rates=points,
            route_mismatches=sum(r['control']['blocked']!=r['expected_route'] for r in results),
            fresh_mismatches={k:sum(r['fresh'][k] is False for r in results) for k in results[0]['fresh']},
            cached_comparisons={k:dict(available=sum(r['cached'].get(k) is not None for r in results),
                                       mismatches=sum(r['cached'].get(k) is False for r in results))
                                for k in ['answer','generated_token_ids','first_logits_sha256','first_top10']},
            speedup={m:averages[m]['latency_ms']['mean']/averages['rollback']['latency_ms']['mean']
                     for m in ['cad','interaction']}))
    finally:
        rollback.close()
        reference.close()
        collector.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',choices=['qwen3b','qwen7b'],required=True)
    main(parser.parse_args().model)
