"""Real validation generation for CPU-shortlisted early gates; no test inputs."""
from __future__ import annotations
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import softmax,expit,logit
from sklearn.model_selection import GroupKFold
from .common import ROOT,STATES,dump,exact_match,is_refusal,read_jsonl,seed_all,write_jsonl
from .metrics import summarize,binary_calibration
from .calibration import fit_temperature,evaluate_calibration,raw_confidence,utility_correct
from .model_runtime import Runtime,Capture
from .early_controller import FoldedGate,EarlyController
from .paired_control import PairedOracleController


def actual(runtime,row,controller=None,capture=None):
    torch.cuda.reset_peak_memory_stats()
    result=runtime.generate(row['question'],row['context'],controller=controller,capture=capture)
    item={k:row[k] for k in ['id','fact_id','state','split','split_id','gold_answers']}
    item.update(result)
    if controller:
        audit=controller.audit();item['control']=audit
        assert audit['gate_calls']==1 and audit['middle_calls']==result['generated_tokens']
        assert audit['query_lengths']==[result['prompt_tokens']]+[1]*(result['generated_tokens']-1)
        blocked=audit.get('block',audit['route']==1)
        expected=(len(runtime.model.model.layers)-audit['gate_layer'])*result['generated_tokens'] if blocked else 0
        assert audit['attention_calls']==expected and audit['residual_edits']==0
    assert item['forward_calls']==item['generated_tokens']
    return item


def refusal(item,route,probability):
    item=dict(item)
    item['automatic_route']=int(route);item['route_probabilities']=list(map(float,probability))
    if int(route)==3:
        item.update(raw_generated_answer=item['answer'],answer='I do not know',refusal_policy='post_generation_replace')
        # The probability remains the empirical probe posterior; costs only choose actions.
        item['control']={**item.get('control',{}),'route_probabilities':item['route_probabilities']}
    return item


def full_summary(rows):
    result=summarize(rows)
    spec=fit_temperature(rows)
    result['answer_utility_calibration_apparent_validation']=evaluate_calibration(rows,spec)
    result['answer_temperature_fit']=spec
    result['answer_utility_raw']=binary_calibration([utility_correct(r) for r in rows],[raw_confidence(r) for r in rows])
    # This still belongs to validation, but each temperature excludes the held-out fact group.
    p=np.empty(len(rows));folds=[]
    for train,test in GroupKFold(n_splits=5).split(rows,groups=[r['fact_id'] for r in rows]):
        fitted=fit_temperature([rows[i] for i in train]);folds.append({'temperature':fitted['temperature'],'held_out_ids':[rows[i]['id'] for i in test]})
        p[test]=expit(logit(np.clip([raw_confidence(rows[i]) for i in test],1e-6,1-1e-6))/fitted['temperature'])
    result['answer_utility_crossfit_validation']=binary_calibration([utility_correct(r) for r in rows],p)
    result['temperature_crossfit_folds']=folds
    if all('route_probabilities' in r for r in rows):
        probabilities=np.array([r['route_probabilities'] for r in rows])
        y=np.array([STATES.index(r['state']) for r in rows])
        result['route_multiclass_brier']=float(np.mean(np.sum((probabilities-np.eye(4)[y])**2,axis=1)))
        result['route_param_calibration']=binary_calibration(y==1,probabilities[:,1])
        result['ext_blocked']=int(sum(r['state']=='ext' and r['automatic_route']==1 for r in rows))
        result['ext_block_rate']=result['ext_blocked']/result['ext_n']
        result['routing_confusion']=[[int(sum(r['state']==STATES[i] and r['automatic_route']==j for r in rows)) for j in range(4)] for i in range(4)]
    return result


def benchmark_gate(gate,seed=42):
    torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    values=[torch.randn((1,gate.weight.shape[0]//len(gate.layers)),device=gate.weight.device,dtype=torch.float16) for _ in gate.layers]
    with torch.inference_mode():
        for _ in range(100):gate.decide(values)
        device=[];host=[]
        for _ in range(10):
            start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True)
            torch.cuda.synchronize();t=time.perf_counter();start.record()
            for _ in range(100):gate.decide(values)
            end.record();torch.cuda.synchronize()
            device.append(start.elapsed_time(end)/100);host.append((time.perf_counter()-t)*10)
        dispatch=[]
        for _ in range(100):
            torch.cuda.synchronize();t=time.perf_counter();route,_,_=gate.decide(values);bool((route==1).item())
            dispatch.append((time.perf_counter()-t)*1000)
    return {'warmup':100,'batches':10,'repeats_per_batch':100,'device_ms_mean':float(np.mean(device)),
        'device_ms_sd':float(np.std(device)),'host_amortized_ms_mean':float(np.mean(host)),
        'with_host_decision_ms_mean':float(np.mean(dispatch)),'with_host_decision_ms_sd':float(np.std(dispatch)),
        'scope':'concatenate available last-token states + one folded addmm + softmax + cost/threshold; host decision separate',
        'target_under_0_1ms_device':bool(np.mean(device)<.1), 'target_under_0_1ms_host_decision':bool(np.mean(dispatch)<.1)}


def main(cpu,output):
    cpu=Path(cpu);output=Path(output);output.mkdir(parents=True,exist_ok=True)
    protocol=json.loads((cpu/'protocol.json').read_text());seeds=seed_all(protocol['seed'])
    source=read_jsonl(cpu/'native.jsonl');validation=[r for r in source if r['split']=='val']
    assert len(validation)==242 and not any(r['split']=='test' for r in source)
    shortlist=json.loads((cpu/'shortlist.json').read_text())['candidates']
    usage=json.loads((cpu/'data_usage.json').read_text());assert usage['validation_ids']==[r['id'] for r in validation]
    signature={'protocol':protocol,'shortlist_sha256':hashlib.sha256((cpu/'shortlist.json').read_bytes()).hexdigest(),
        'seed_state':seeds,'validation_ids':usage['validation_ids'],'test_generations':0,
        'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}}
    if (output/'signature.json').exists():assert json.loads((output/'signature.json').read_text())==signature
    else:dump(output/'signature.json',signature)
    runtime=Runtime(max_input=protocol['max_input_tokens'],max_new=protocol['max_new_tokens'])
    for _ in range(3):runtime.generate('What is the capital of France?','The capital of France is Paris.')
    base_path=output/'vanilla.jsonl';base=read_jsonl(base_path) if base_path.exists() else []
    feature_dir=output/'baseline_features';feature_dir.mkdir(exist_ok=True)
    capture=Capture(runtime.model)
    try:
        for row in validation[len(base):]:
            item=actual(runtime,row,capture=capture);base.append(item)
            np.save(feature_dir/(row['id']+'.npy'),capture.array())
            with base_path.open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
    finally:capture.close()
    assert [r['id'] for r in base]==usage['validation_ids']
    baseline=summarize(base)
    by_id={r['id']:r for r in base};branch={}
    # All five Oracle references use actual param generations and identity on other states.
    oracles={}
    dummy={'mean_delta':np.zeros(runtime.model.config.hidden_size),'cb_center':np.zeros(runtime.model.config.hidden_size),'basis':{}}
    for layer in protocol['layers']:
        path=output/f'oracle_layer{layer}.jsonl';rows=[]
        for row in validation:
            if row['state']=='param':
                controller=PairedOracleController(runtime.model,layer,dummy,span_mode='hard',attention_scope='all_queries',oracle_route=1)
                try:item=actual(runtime,row,controller)
                finally:controller.close()
                branch[(layer,row['id'])]=item
            else:item=by_id[row['id']]
            rows.append(refusal(item,STATES.index(row['state']),np.eye(4)[STATES.index(row['state'])]))
        write_jsonl(path,rows);oracles[str(layer)]=summarize(rows)
        print({'oracle_layer':layer,'param_correct':round(oracles[str(layer)]['param_accuracy']*13)},flush=True)
    summaries=[];composed={};gates={}
    for candidate in shortlist:
        key=candidate['key'];file=cpu/(key+'.npz')
        assert hashlib.sha256(file.read_bytes()).hexdigest()==candidate['artifact_sha256']
        gate=FoldedGate(file,candidate['policy'],next(runtime.model.parameters()).device);gates[key]=gate
        # Same pre-control hidden states, obtained during baseline's ordinary prefill.
        expected=[];probabilities=[]
        with torch.inference_mode():
            for row in validation:
                cached=np.load(feature_dir/(row['id']+'.npy'))
                values=[torch.as_tensor(cached[l-1:l],device=gate.weight.device,dtype=torch.float16) for l in gate.layers]
                route,prob,_=gate.decide(values);expected.append(int(route.item()));probabilities.append(prob[0].float().cpu().numpy())
        candidate['cloud_routes']=expected
        rows=[]
        for i,row in enumerate(validation):
            if expected[i]==1:
                cache_key=(candidate['spec']['layer'],row['id'])
                if cache_key not in branch:
                    controller=EarlyController(runtime.model,gate)
                    try:item=actual(runtime,row,controller)
                    finally:controller.close()
                    branch[cache_key]=item
                item=branch[cache_key]
            else:item=by_id[row['id']]
            item=refusal(item,expected[i],probabilities[i]);item['quality_source']='real generated branch, shared for identical layer/action/input'
            rows.append(item)
        composed[key]=rows
        summary=full_summary(rows)
        eligible=(summary['ext_accuracy']>=baseline['ext_accuracy']-protocol['max_ext_accuracy_drop']-1e-12 and
            summary['align_accuracy']>=baseline['align_accuracy']-1e-12 and summary['ext_blocked']<=protocol['max_ext_blocked'])
        summaries.append({'key':key,'spec':candidate['spec'],'policy':candidate['policy'],'eligible':eligible,
            'composition_summary':summary})
        write_jsonl(output/(key+'_composed.jsonl'),rows)
        print({'candidate':key,'layer':candidate['spec']['layer'],'h':summary['h_score'],'eligible':eligible},flush=True)
    # A complete fresh 242-generation pass per winning layer verifies composition,
    # obtains genuine per-request timings and measures actual in-flight probabilities.
    winners=[]
    def objective(item):
        s=item.get('actual_summary',item['composition_summary'])
        return (item['eligible'],s['h_score'],s['refusal_f1'],s['align_accuracy'],-s['ext_blocked'],-len(item['spec']['layers']))
    for layer in protocol['layers']:
        pool=[s for s in summaries if s['spec']['layer']==layer]
        winner=max(pool,key=objective);key=winner['key'];controller=EarlyController(runtime.model,gates[key])
        actual_rows=[];differences=[]
        try:
            for i,row in enumerate(validation):
                item=actual(runtime,row,controller);audit=item['control']
                item=refusal(item,audit['route'],audit['route_probabilities']);actual_rows.append(item)
                prior=composed[key][i]
                if item['answer']!=prior['answer'] or item['automatic_route']!=prior['automatic_route']:
                    differences.append({'id':row['id'],'cpu_route':prior['automatic_route'],'online_route':item['automatic_route'],
                        'cache_answer':prior['answer'],'online_answer':item['answer']})
                if (i+1)%40==0:print({'measured_layer':layer,'completed':i+1,'total':242},flush=True)
        finally:controller.close()
        write_jsonl(output/(key+'_actual.jsonl'),actual_rows)
        winner['actual_summary']=full_summary(actual_rows)
        s=winner['actual_summary'];winner['eligible']=(s['ext_accuracy']>=baseline['ext_accuracy']-protocol['max_ext_accuracy_drop']-1e-12 and
            s['align_accuracy']>=baseline['align_accuracy']-1e-12 and s['ext_blocked']<=protocol['max_ext_blocked'])
        winner['composition_online_differences']=differences
        winner['gate_benchmark']=benchmark_gate(gates[key])
        winners.append(winner);dump(output/'progress.json',{'winners':winners})
    selected=max(winners,key=objective)
    # Matched 12-token workload on fixed validation facts, with interleaved Vanilla/CAD/selected.
    rng=np.random.default_rng(42);ids=rng.choice(len(validation),8,replace=False).tolist()
    fixed=[];runtime.fixed_length_benchmark=True
    controller=EarlyController(runtime.model,gates[selected['key']])
    try:
        for rep in range(3):
            for i in ids:
                for method in ['vanilla','cad','early_gate']:
                    torch.cuda.reset_peak_memory_stats()
                    result=runtime.generate(validation[i]['question'],validation[i]['context'],
                        controller=controller if method=='early_gate' else None,cad_alpha=1. if method=='cad' else None)
                    assert result['forward_calls']==12*(2 if method=='cad' else 1)
                    fixed.append({'id':validation[i]['id'],'rep':rep,'method':method,**result})
    finally:controller.close();runtime.fixed_length_benchmark=False
    write_jsonl(output/'fixed_workload.jsonl',fixed)
    timing={m:{'n':24,'latency_mean_ms':float(np.mean([r['latency_ms'] for r in fixed if r['method']==m])),
        'latency_sd_ms':float(np.std([r['latency_ms'] for r in fixed if r['method']==m])),
        'peak_allocated_bytes':max(r['peak_allocated_bytes'] for r in fixed if r['method']==m)} for m in ['vanilla','cad','early_gate']}
    timing['speedup_vs_cad']=timing['cad']['latency_mean_ms']/timing['early_gate']['latency_mean_ms']
    timing['vram_overhead_bytes']=timing['early_gate']['peak_allocated_bytes']-timing['vanilla']['peak_allocated_bytes']
    dump(output/'summary.json',{'complete':True,'validation_n':242,'test_generations':0,'baseline':baseline,
        'oracle_by_layer':oracles,'candidates':summaries,'layer_winners':winners,'selected':selected,'fixed_workload':timing,
        'selection_scope':'validation tuning result, not independent confirmation; 13 param facts',
        'confidence_scope':'raw/fitted validation and grouped crossfit utility calibration; costs do not rewrite empirical probe posterior'})
    print({'complete':True,'selected_layer':selected['spec']['layer'],'h':selected['actual_summary']['h_score']},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--cpu',default='results/early_gate_cpu_v1');p.add_argument('--output',default='results/early_gate_v1')
    a=p.parse_args();main(a.cpu,a.output)
