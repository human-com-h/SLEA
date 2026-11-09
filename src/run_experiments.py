"""Validation selection followed by a frozen, independent test evaluation."""
from __future__ import annotations
import argparse
import json
import hashlib
import time
from pathlib import Path
import numpy as np
import torch
from .common import ROOT,SEEDS,STATES,dump,exact_match,read_jsonl,seed_all,write_jsonl,cuda_sync,fingerprint
from .manifold_arbitrator import ManifoldArbitrator
from .model_runtime import Runtime
from .slea_controller import SLEAController
from .metrics import summarize
from .output_gates import InteractionScore,select_thresholds,threshold_value
from .calibration import fit_temperature,evaluate_calibration

METHODS=['vanilla','closed_book','entropy','margin','interaction','channel1','channel2',
         'channel1_tuned','channel2_tuned','dual','static_span','oracle','random','cad']

def trial_objective(trial):
    s=trial['summary'];c=trial['config']
    return (-(s['h_score'] or 0),-s['refusal_f1'],-(s['align_accuracy'] or 0),c['span_bias'],c['residual_strength'])

def run(data,output,seed=42,max_new=12,max_input=512,stage='val',limit=None):
    seed_all(seed)
    data,output=Path(data),Path(output)
    offline=ManifoldArbitrator.load(output/'arbitrator.npz')
    meta=json.loads((output/'arbitrator.npz.json').read_text(encoding='utf-8'))['metadata']
    layer=meta['selected']['layer']
    rows=[r for r in read_jsonl(data/'native.jsonl') if r['split']==stage]
    if limit:rows=rows[:limit]
    if not rows:raise ValueError('No evaluation samples')
    if any(r['split_id']!=meta['split_id'] for r in rows):raise ValueError('Partition mismatch')
    frozen_path=output/'frozen_generation.json'
    if stage=='test' and not frozen_path.exists():raise ValueError('Validation must freeze generation configuration first')
    runtime=Runtime(max_input=max_input,max_new=max_new)
    online=offline.deploy()
    runtime.generate('What is the capital of France?')
    with torch.inference_mode():
        for _ in range(3):online(online.center[None,:])
    if stage=='val':
        vanilla=evaluate_method(runtime,online,layer,rows,'vanilla',output/'val',{},seed)
        cb={r['fact_id']:r for r in read_jsonl(data/'closed_book.jsonl')}
        encoder=InteractionScore()
        thresholds,gate_search=select_thresholds(vanilla,cb,encoder,rows)
        dump(output/'output_gate_validation.json',gate_search)
        del encoder
        torch.cuda.empty_cache()
        routing=json.loads((output/'routing_weight_validation.json').read_text())
        weights=sorted({1.,routing['chosen']['prior_weight']})
        configurations=[{'residual_strength':a,'span_bias':b,'prior_weight':w}
                        for w in weights for a in [0.,.5,1.] for b in [0.,1.,3.]]
        trials=[]
        # Validation controls may be expensive; all results are preserved and resumable.
        for cfg in configurations:
            method=f"dual_eta{cfg['residual_strength']}_beta{cfg['span_bias']}_weight{cfg['prior_weight']}"
            results=evaluate_method(runtime,online,layer,rows,method,output/'val',cfg,seed)
            summary=summarize(results)
            trials.append({'config':cfg,'summary':summary,'method':method})
        selected=min(trials,key=trial_objective)
        method_configs={m:selected['config'] for m in METHODS}
        method_configs['channel1_tuned']=min([t for t in trials if t['config']['span_bias']==0],key=trial_objective)['config']
        method_configs['channel2_tuned']=min([t for t in trials if t['config']['residual_strength']==0],key=trial_objective)['config']
        static_trials=[]
        for beta in [0.,1.,3.]:
            cfg={'residual_strength':0.,'span_bias':beta}
            result=evaluate_method(runtime,online,layer,rows,f'static_span__beta{beta}',output/'val',cfg,seed)
            static_trials.append({'config':cfg,'summary':summarize(result)})
        method_configs['static_span']=min(static_trials,key=trial_objective)['config']
        dump(output/'static_span_validation_search.json',static_trials)
        calibration={}
        for method in METHODS:
            result=evaluate_method(runtime,online,layer,rows,method,output/'val',method_configs[method],seed,thresholds)
            calibration[method]=fit_temperature(result)
        dump(output/'validation_generation_search.json',trials)
        dump(frozen_path,{'seed':seed,'split_id':meta['split_id'],'layer':layer,'config':selected['config'],
            'max_new_tokens':max_new,'max_input_tokens':max_input,'validation_ids':[r['id'] for r in rows],
            'output_gate_thresholds':thresholds,
            'method_configs':method_configs,'answer_utility_calibration':calibration,
            'selection_rule':'max H-score, then refusal F1, align accuracy, smaller span bias'})
        return
    frozen=json.loads(frozen_path.read_text(encoding='utf-8'))
    if frozen['max_new_tokens']!=max_new or frozen['max_input_tokens']!=max_input:
        raise ValueError('Generation budget differs from frozen validation')
    cfg=frozen['config']
    methods=METHODS
    summaries={}
    for method in methods:
        result=evaluate_method(runtime,online,layer,rows,method,output/'test',frozen['method_configs'][method],seed,
                               thresholds=frozen['output_gate_thresholds'])
        summaries[method]=summarize(result)
        summaries[method]['answer_utility_calibration']=evaluate_calibration(result,frozen['answer_utility_calibration'][method])
        chosen=np.random.default_rng(seed).permutation(len(rows))[:min(8,len(rows))]
        timed_rows=[rows[i] for i in chosen]
        repeats=[]
        runtime.fixed_length_benchmark=True
        try:
            for repeat in range(3):
                timed=evaluate_method(runtime,online,layer,timed_rows,method,
                    output/'fixed_workload'/f'rep{repeat}',frozen['method_configs'][method],seed,
                    thresholds=frozen['output_gate_thresholds'])
                repeats.append([{'id':r['id'],'latency_ms':r['latency_ms'],'ttft_ms':r['ttft_ms'],
                    'generated_tokens':r['generated_tokens'],'forward_calls':r['forward_calls']} for r in timed])
        finally:runtime.fixed_length_benchmark=False
        latencies=np.array([[r['latency_ms'] for r in repeat] for repeat in repeats])
        summaries[method]['fixed_workload_timing']={'sample_ids':[r['id'] for r in timed_rows],
            'repeats':3,'tokens_per_branch':max_new,'timing_only':True,
            'mean_ms':float(latencies.mean()),'std_ms':float(latencies.std()),
            'mean_within_sample_std_ms':float(latencies.std(axis=0).mean()),'records':repeats}
        print({'method':method,**{k:v for k,v in summaries[method].items() if k in ['h_score','refusal_f1','latency_mean_ms']}},flush=True)
    cad=summaries['cad']['latency_mean_ms']
    for s in summaries.values():
        s['speedup_vs_cad']=cad/s['latency_mean_ms']
        s['fixed_workload_speedup_vs_cad']=summaries['cad']['fixed_workload_timing']['mean_ms']/s['fixed_workload_timing']['mean_ms']
    dump(output/'main_results.json',summaries)
    columns=['Method','H-Score','ext Acc','param Acc','align Acc','Refusal F1','Latency ms','Speedup/CAD']
    table=['| '+' | '.join(columns)+' |','|'+'---|'*len(columns)]
    for name,s in summaries.items():
        keys=['h_score','ext_accuracy','param_accuracy','align_accuracy','refusal_f1','latency_mean_ms','speedup_vs_cad']
        table.append('| '+name+' | '+' | '.join('NA' if s[k] is None else f'{s[k]:.4f}' for k in keys)+' |')
    (output/'main_results.md').write_text('\n'.join(table)+'\n',encoding='utf-8')

def evaluate_method(runtime,online,layer,rows,method,directory,cfg,seed,thresholds=None):
    directory.mkdir(parents=True,exist_ok=True)
    path=directory/(method+'.jsonl')
    role=method.split('__')[0].replace('_tuned','')
    base_role=role in ['vanilla','closed_book','cad','entropy','margin','interaction']
    if base_role:cfg={}
    online.set_prior_weight(cfg.get('prior_weight',1.))
    intervention_cfg={k:v for k,v in cfg.items() if k in ['residual_strength','span_bias']}
    relevant_thresholds={role:thresholds[role]} if role in ['entropy','margin','interaction'] else None
    arb_path=next(parent/'arbitrator.npz' for parent in [directory,*directory.parents] if (parent/'arbitrator.npz').is_file())
    signature={'method':method,'config':cfg,'seed':seed,'layer':layer,'thresholds':relevant_thresholds,
        'max_new':runtime.max_new,'max_input':runtime.max_input,'split_ids':sorted({r['split_id'] for r in rows}),
        'sample_ids':[r['id'] for r in rows],
        'fixed_length_benchmark':getattr(runtime,'fixed_length_benchmark',False),
        'arbitrator_sha256':hashlib.sha256(arb_path.read_bytes()).hexdigest(),
        'code_hashes':{name:hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest() for name in
                       ['model_runtime.py','slea_controller.py','manifold_arbitrator.py','output_gates.py','calibration.py']}}
    signature_path=directory/(method+'.signature.json')
    if signature_path.exists():
        if json.loads(signature_path.read_text(encoding='utf-8'))!=signature:raise ValueError('Incompatible cached evaluation configuration')
    elif path.exists():raise ValueError('Evaluation cache has no configuration signature')
    else:dump(signature_path,signature)
    old={r['id']:r for r in read_jsonl(path)} if path.exists() else {}
    rng=np.random.default_rng(seed)
    random_routes={r['id']:int(rng.integers(0,4)) for r in rows}
    torch.cuda.reset_peak_memory_stats()
    encoder=InteractionScore() if role=='interaction' else None
    if len(old)<len(rows):
        for warm_route in [0,1]:
            warm=None
            if not base_role:
                warm=SLEAController(runtime.model,online,layer,
                    channel1=role not in ['channel2','static_span'],channel2=role!='channel1',
                    force_route=warm_route,**intervention_cfg)
            try:
                runtime.generate('What is the capital of France?',None if role=='closed_book' else 'The capital of France is Paris.',
                    controller=warm,cad_alpha=1. if role=='cad' else None)
            finally:
                if warm:warm.close()
    for i,row in enumerate(rows):
        if row['id'] in old:continue
        control=None
        if role not in ['vanilla','closed_book','cad','entropy','margin','interaction']:
            force=STATES.index(row['state']) if role=='oracle' else random_routes[row['id']] if role=='random' else 1 if role=='static_span' else None
            control=SLEAController(runtime.model,online,layer,
                channel1=role not in ['channel2','static_span'],channel2=role!='channel1',
                force_route=force,**intervention_cfg)
        try:
            if role=='interaction':
                cuda_sync();begin=time.perf_counter()
                cb=runtime.generate(row['question'])
                score,encoding_ms=encoder.score(cb['answer'],row['context'],cb['first_token']['confidence'])
                choose_prior=score>thresholds['interaction']
                result=dict(cb) if choose_prior else runtime.generate(row['question'],row['context'])
                cuda_sync();total=(time.perf_counter()-begin)*1000
                result['branch_latency_ms']=result['latency_ms']
                result['latency_ms']=total
                result['ttft_ms']=total if choose_prior else cb['latency_ms']+encoding_ms+result['ttft_ms']
                if not choose_prior:result['forward_calls']+=cb['forward_calls']
                result.update(interaction_score=score,sentence_encoder_ms=encoding_ms,
                    closed_book_ms=cb['latency_ms'],closed_book_generated_tokens=cb['generated_tokens'],
                    source_selected='closed_book' if choose_prior else 'rag',
                    total_model_generated_tokens=cb['generated_tokens']+(0 if choose_prior else result['generated_tokens']))
            else:
                result=runtime.generate(row['question'],None if role=='closed_book' else row['context'],
                    controller=control,cad_alpha=1. if role=='cad' else None)
            if role in ['entropy','margin']:
                value=result['first_token']['entropy' if role=='entropy' else 'margin']
                threshold=threshold_value(thresholds[role])
                refused=value>threshold if role=='entropy' else value<threshold
                if refused:
                    result['raw_generated_answer']=result['answer'];result['answer']='I do not know'
                    result['refusal_policy']='post_generation_replace'
            if control:
                result['control']=control.audit()
                assert result['control']['gate_calls']==1
                assert result['forward_calls']==result['generated_tokens']
                if result['control']['route']==3:
                    result['raw_generated_answer']=result['answer']
                    result['answer']='I do not know'
                    result['refusal_policy']='post_generation_replace'
                    # Generation budget/calls are retained, avoiding an early-stop speed claim.
        finally:
            if control:control.close()
        item={k:row[k] for k in ['id','fact_id','split','split_id','state','gold_answers']}
        item.update(result)
        item.update(method=method,seed=seed)
        with path.open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
        old[item['id']]=item
        if (i+1)%20==0:print({'method':method,'completed':i+1,'total':len(rows)},flush=True)
    if encoder:
        del encoder
        torch.cuda.empty_cache()
    return [old[r['id']] for r in rows]

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/processed/conflictqa_qwen25_3b_v1')
    p.add_argument('--output',default='results/qwen25_3b_seed42')
    p.add_argument('--seed',type=int,default=42,choices=SEEDS)
    p.add_argument('--stage',choices=['val','test'],default='val')
    p.add_argument('--max-new',type=int,default=12)
    p.add_argument('--max-input',type=int,default=512)
    p.add_argument('--limit',type=int)
    a=p.parse_args();run(a.data,a.output,a.seed,a.max_new,a.max_input,a.stage,a.limit)
