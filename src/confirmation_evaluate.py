"""One frozen blind trajectory per sample/method; no fitting on confirmation/test."""
from __future__ import annotations
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit,logit
from .common import ROOT,STATES,dump,read_jsonl,seed_all,write_jsonl,cuda_sync
from .model_runtime import Runtime
from .early_controller import FoldedGate,EarlyController
from .paired_control import PairedOracleController
from .slea_controller import SLEAController
from .output_gates import InteractionScore,threshold_value
from .calibration import raw_confidence


class LastQueryController(EarlyController):
    def _attention(self,module,args,kwargs):
        if self.span is None or not self.block:return args,kwargs
        mask=kwargs.get('attention_mask')
        if mask is None or mask.ndim!=4 or mask.dtype!=torch.float16:raise ValueError('Explicit fp16 mask required')
        start,end=self.span;assert 0<=start<end<=mask.shape[-1]
        changed=mask.clone();changed[:,:,-1,start:end]=-float('inf')
        kwargs=dict(kwargs);kwargs['attention_mask']=changed;self.attention_calls+=1
        return args,kwargs

    def audit(self):
        result=super().audit();result.update(attention_scope='last_query',
            hard_mask_scope='upper-layer final prompt query and ordinary decode query to context keys')
        return result


def attach_confidence(item,method,protocol):
    raw=float(np.clip(raw_confidence(item),1e-6,1-1e-6))
    if method in ['early_gate','last_query','oracle','static_hard']:
        temps=protocol['fold_temperatures']
    else:temps=[protocol['baseline_temperatures'][method]]
    item['utility_confidence_raw']=raw
    item['utility_confidence_frozen']=float(np.mean([expit(logit(raw)/t) for t in temps]))
    item['confidence_temperatures']=temps
    if method in ['early_gate','last_query']:
        item['utility_confidence_global_T2']=float(expit(logit(raw)/protocol['global_temperature']))
    return item


def controller_for(runtime,gate,method,protocol):
    if method=='early_gate':return EarlyController(runtime.model,gate)
    if method=='last_query':return LastQueryController(runtime.model,gate)
    if method in ['oracle','static_hard']:
        d=runtime.model.config.hidden_size
        return PairedOracleController(runtime.model,10,{'mean_delta':np.zeros(d),'cb_center':np.zeros(d),'basis':{}},
            span_mode='hard',attention_scope='all_queries',oracle_route=1)
    if method=='static_span':
        cfg=protocol['static_span']
        return SLEAController(runtime.model,None,cfg['layer'],channel1=False,channel2=True,
            residual_strength=0.,span_bias=cfg['span_bias'],force_route=1)
    return None


def request(runtime,row,method,protocol,controller=None,encoder=None):
    torch.cuda.reset_peak_memory_stats()
    if method=='interaction':
        cuda_sync();start=time.perf_counter()
        cb=runtime.generate(row['question'])
        score,ms=encoder.score(cb['answer'],row['context'],cb['first_token']['confidence'])
        prior=score>protocol['baseline_thresholds']['interaction']
        result=dict(cb) if prior else runtime.generate(row['question'],row['context'])
        cuda_sync();total=(time.perf_counter()-start)*1000
        result.update(branch_latency_ms=result['latency_ms'],latency_ms=total,
            ttft_ms=total if prior else cb['latency_ms']+ms+result['ttft_ms'],
            interaction_score=score,sentence_encoder_ms=ms,closed_book_ms=cb['latency_ms'],
            closed_book_generated_tokens=cb['generated_tokens'],source_selected='closed_book' if prior else 'rag',
            total_model_generated_tokens=cb['generated_tokens']+(0 if prior else result['generated_tokens']))
        if not prior:result['forward_calls']+=cb['forward_calls']
        assert result['forward_calls']==result['total_model_generated_tokens']
    else:
        if method=='oracle':controller.force_route=STATES.index(row['state'])
        result=runtime.generate(row['question'],row['context'],controller=controller,
            cad_alpha=protocol['cad_alpha'] if method=='cad' else None)
        assert result['forward_calls']==result['generated_tokens']*(2 if method=='cad' else 1)
        if controller:
            control=controller.audit();result['control']=control
            assert control['gate_calls']==1 and control['middle_calls']==result['generated_tokens']
            assert control['query_lengths']==[result['prompt_tokens']]+[1]*(result['generated_tokens']-1)
            assert control['residual_edits']==0
            if method=='static_hard':control.update(oracle=False,active_only_on_param=False,static=True)
            if method in ['early_gate','last_query','oracle','static_hard']:
                block=control['route']==1
                assert control['attention_calls']==26*result['generated_tokens']*block
            if method in ['early_gate','last_query','oracle'] and control['route']==3:
                result.update(raw_generated_answer=result['answer'],answer='I do not know',refusal_policy='post_generation_replace')
        if method in ['entropy','margin']:
            signal=result['first_token']['entropy' if method=='entropy' else 'margin']
            threshold=threshold_value(protocol['baseline_thresholds'][method])
            if (signal>threshold) if method=='entropy' else (signal<threshold):
                result.update(raw_generated_answer=result['answer'],answer='I do not know',refusal_policy='post_generation_replace')
    item={k:row[k] for k in ['id','fact_id','component_id','split','split_id','state','gold_answers','prior_correct'] if k in row}
    item.update(result,method=method,seed=42)
    return attach_confidence(item,method,protocol)


def evaluate(runtime,rows,method,protocol,gate,directory,signature):
    directory.mkdir(exist_ok=True);path=directory/(method+'.jsonl');sign=directory/(method+'.signature.json')
    spec={**signature,'method':method,'ids':[r['id'] for r in rows]}
    if sign.exists():assert json.loads(sign.read_text())==spec
    else:
        assert not path.exists();dump(sign,spec)
    old=read_jsonl(path) if path.exists() else [];assert [r['id'] for r in old]==spec['ids'][:len(old)]
    control=controller_for(runtime,gate,method,protocol);encoder=InteractionScore() if method=='interaction' else None
    try:
        warm={'question':'What is the capital of France?','context':'The capital of France is Paris.','state':'param',
            'id':'warmup','fact_id':'warmup','split':'warmup','split_id':'warmup','gold_answers':['Paris'],'prior_correct':True}
        for _ in range(2):request(runtime,warm,method,protocol,control,encoder)
        for i,row in enumerate(rows[len(old):],start=len(old)):
            item=request(runtime,row,method,protocol,control,encoder)
            with path.open('a',encoding='utf-8') as f:f.write(json.dumps(item,ensure_ascii=False)+'\n')
            old.append(item)
            if (i+1)%30==0:print({'split':row['split'],'method':method,'completed':i+1,'total':len(rows)},flush=True)
    finally:
        if control:control.close()
        if encoder:del encoder;torch.cuda.empty_cache()
    return old


def fixed_timing(runtime,rows,gate,protocol,output):
    rng=np.random.default_rng(42);selected=[]
    for state in STATES:
        group=[r for r in rows if r['state']==state]
        selected += [group[i] for i in rng.choice(len(group),2,replace=False)]
    methods=['vanilla','cad','early_gate','last_query','interaction'];records=[]
    encoder=InteractionScore();control=EarlyController(runtime.model,gate);last=LastQueryController(runtime.model,gate)
    runtime.fixed_length_benchmark=True
    try:
        for rep in range(3):
            for i,row in enumerate(selected):
                shift=(rep+i)%len(methods);order=methods[shift:]+methods[:shift]
                for method in order:
                    control.reset(None);last.reset(None)
                    c=control if method=='early_gate' else last if method=='last_query' else None
                    item=request(runtime,row,method,protocol,c,encoder);item['rep']=rep
                    assert item['generated_tokens']==12
                    records.append(item)
    finally:control.close();last.close();runtime.fixed_length_benchmark=False;del encoder
    write_jsonl(output/'fixed_workload.jsonl',records)


def main():
    protocol=json.loads((ROOT/'configs/confirmation_v1.json').read_text());seed_state=seed_all(42)
    for name,sha in protocol['frozen_code_sha256'].items():assert hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()==sha
    weights=ROOT/'results/early_gate_frozen_v1.npz';assert hashlib.sha256(weights.read_bytes()).hexdigest()==protocol['weights_sha256']
    locked=ROOT/'results/confirmation_prescreen_v1';lock=json.loads((locked/'sample_lock.json').read_text())
    assert lock['locked_before_slea'] and lock['slea_generations_before_lock']==0
    assert hashlib.sha256((locked/'locked_native.jsonl').read_bytes()).hexdigest()==lock['files']['locked_native.jsonl']
    assert hashlib.sha256((ROOT/'configs/confirmation_v1.json').read_bytes()).hexdigest()==lock['configuration_sha256']
    rows=read_jsonl(locked/'locked_native.jsonl');assert len(rows)==300 and len({r['fact_id'] for r in rows})==150
    output=ROOT/'results/confirmation_evaluation_v1';output.mkdir(parents=True,exist_ok=True)
    signature={'protocol':protocol,'sample_lock_sha256':hashlib.sha256((locked/'sample_lock.json').read_bytes()).hexdigest(),
        'seed_state':seed_state,'code_sha256':{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in (ROOT/'src').glob('*.py')}}
    dump(output/'evaluation_lock.json',signature)
    runtime=Runtime(max_input=512,max_new=12);gate=FoldedGate(weights,protocol['policy'],'cuda:0')
    for method in protocol['methods']:evaluate(runtime,rows,method,protocol,gate,output/'confirmation',signature)
    old=read_jsonl(ROOT/'data/processed/confirmation_v1/old_test_inputs.jsonl');assert len(old)==254
    evaluate(runtime,old,'early_gate',protocol,gate,output/'old_test',signature)
    fixed_timing(runtime,rows,gate,protocol,output)
    dump(output/'completion.json',{'complete':True,'new_contexts':300,'new_facts':150,'old_test_rows':254,
        'methods':protocol['methods'],'parameter_refits':0,'sample_replacements_after_lock':0,
        'signature_sha256':hashlib.sha256((output/'evaluation_lock.json').read_bytes()).hexdigest()})
    print('Frozen confirmation and old-test evaluation complete',flush=True)


if __name__=='__main__':main()
