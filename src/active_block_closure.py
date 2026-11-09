"""Frozen 3B cache completion and matched active-control timing; no fitting."""
import argparse
import hashlib
import json
import time
from pathlib import Path
from datetime import datetime, timezone
import numpy as np
import torch
from .common import ROOT, dump, read_jsonl, seed_all
from .early_controller import FoldedGate, EarlyController
from .confirmation_evaluate import LastQueryController, request as request3
from .phase4_runtime import Phase4Runtime, Phase4Gate, ScopeController, synchronize
from .phase4_common import request as request7, append_jsonl, attach_probability
from .model_runtime import Runtime
from .output_gates import InteractionScore


class ActiveMixin:
    def decide(self, values):
        original, probability, logits = super().decide(values)
        self.original_route = original.clone()
        active = torch.where((original == 1) | (original == 3), torch.ones_like(original), original)
        return active, probability, logits


class ActiveFoldedGate(ActiveMixin, FoldedGate): pass
class ActivePhase4Gate(ActiveMixin, Phase4Gate): pass


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def allocated():return [torch.cuda.memory_allocated(i) for i in range(torch.cuda.device_count())]
def difference(a,b):return [int(x-y) for x,y in zip(a,b)]


def timing_order(methods, repeat, index):
    offset=(repeat+index)%len(methods)
    return methods[offset:]+methods[:offset]


def original_route(result,method,active_gate):
    return int(active_gate.original_route.item()) if method in ['active_block','last_active_block'] else result.get('control',{}).get('route')


def check_inputs(model):
    cfg=ROOT/'configs/active_block_closure_v1.json'
    inp=ROOT/'data/processed/active_block_closure_v1'/model
    lock=json.loads((inp/'input_lock.json').read_text())
    assert sha(cfg)==lock['protocol_sha256']
    for name,digest in lock['files'].items():assert sha(inp/name)==digest
    if model=='qwen3b':
        protocol=json.loads((ROOT/'configs/confirmation_v1.json').read_text())
        assert sha(ROOT/'results/early_gate_frozen_v1.npz')==protocol['weights_sha256']
        for name,digest in protocol['frozen_code_sha256'].items():assert sha(ROOT/'src'/name)==digest
    else:
        protocol=json.loads((ROOT/'results/phase4_qwen7b_v1/development/freeze.json').read_text())
        assert sha(ROOT/'results/phase4_qwen7b_v1/development/frozen_gate.npz')==protocol['weights_sha256']
        for name,digest in protocol['source_sha256'].items():
            p=ROOT/'src'/name
            # Packages use exact development-frozen bytes where local reporting evolved.
            if p.exists() and name in ['phase4_runtime.py','phase4_common.py','phase4_numerics.py','model_runtime.py','early_controller.py','slea_controller.py','output_gates.py','common.py']:
                assert sha(p)==digest,name
    return {'model':model,'input_hashes_passed':True,'frozen_core_hashes_passed':True,'gpu_model_calls':0}


def main(model):
    check_inputs(model)
    cfg=json.loads((ROOT/'configs/active_block_closure_v1.json').read_text())
    inp=ROOT/'data/processed/active_block_closure_v1'/model
    lock=json.loads((inp/'input_lock.json').read_text())
    assert sha(ROOT/'configs/active_block_closure_v1.json')==lock['protocol_sha256']
    for name,digest in lock['files'].items():assert sha(inp/name)==digest,name
    out=ROOT/'results/active_block_closure_v1'/model;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'entry_lock.json').exists(),'No automatic repetition of a quality job'
    seeds=seed_all(42)
    dump(out/'entry_lock.json',{'seed_state':seeds,'input_lock':lock,'started_utc':datetime.now(timezone.utc).isoformat(),
        'new_fits':0,'stage2_used':False,'warmup_requests':14,'timing_requests':168,'quality_requests':234 if model=='qwen3b' else 0})
    if model=='qwen3b':
        protocol=json.loads((ROOT/'configs/confirmation_v1.json').read_text())
        for name,digest in protocol['frozen_code_sha256'].items():assert sha(ROOT/'src'/name)==digest
        weights=ROOT/'results/early_gate_frozen_v1.npz';assert sha(weights)==protocol['weights_sha256']
        runtime=Runtime(max_input=512,max_new=12);layer=10
        runtime_cfg=protocol
        plain_class,active_class=FoldedGate,ActiveFoldedGate
    else:
        runtime_cfg=json.loads((ROOT/'configs/phase4_qwen7b_v1.json').read_text())
        protocol=json.loads((ROOT/'results/phase4_qwen7b_v1/development/freeze.json').read_text())
        weights=ROOT/'results/phase4_qwen7b_v1/development/frozen_gate.npz'
        assert sha(weights)==protocol['weights_sha256']
        for name,digest in protocol['source_sha256'].items():
            path=ROOT/'src'/name
            if path.exists():assert sha(path)==digest,name
        runtime=Phase4Runtime(runtime_cfg);layer=11
        plain_class,active_class=Phase4Gate,ActivePhase4Gate
    synchronize();model_only=allocated()
    encoder=InteractionScore();synchronize();after_encoder=allocated()
    device=next(runtime.model.model.layers[layer-1].parameters()).device
    plain=plain_class(weights,protocol['policy'],device);synchronize();after_plain=allocated()
    active=active_class(weights,protocol['policy'],device);synchronize();after_active=allocated()
    # Gates/encoder remain resident, but baselines must not pay inactive SLEA hooks.
    live_controls=[]
    def make_control(method):
        assert not live_controls
        if method not in ['early_gate','last_query','active_block','last_active_block']:return None
        selected_gate=active if method in ['active_block','last_active_block'] else plain
        if model=='qwen3b':
            cls=LastQueryController if method in ['last_query','last_active_block'] else EarlyController
            controller=cls(runtime.model,selected_gate)
        else:
            controller=ScopeController(runtime,layer,scope='last_query' if method in ['last_query','last_active_block'] else 'all_queries',gate=selected_gate)
        live_controls.append(controller)
        return controller
    synchronize();shared=allocated()
    memory={'model_only':model_only,'after_encoder':after_encoder,'after_original_gate':after_plain,
            'after_active_gate':after_active,'shared_ready':shared,
            'encoder_static':difference(after_encoder,model_only),'original_gate_static':difference(after_plain,after_encoder),
            'active_gate_static':difference(after_active,after_plain),
            'note':'common residency for timing; attributable per-method peak is accounting, not separately loaded deployment peak'}
    dump(out/'memory_initialization.json',memory)
    dump(out/'environment.json',{'torch':torch.__version__,'model':model,'layer':layer,'dtype':'float16','backend':'eager',
        'device_map':getattr(runtime.model,'hf_device_map',{}),'gpu_names':[torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
        'all_parameters_frozen':all(not p.requires_grad and p.dtype==torch.float16 for p in runtime.model.parameters()),
        'seed_state':seeds,'new_fits':0,'batch_size':1})

    def request(row,method):
        controller=make_control(method)
        synchronize();idle=allocated()
        for i in range(torch.cuda.device_count()):torch.cuda.reset_peak_memory_stats(i)
        started=time.perf_counter()
        mapped={'active_block':'early_gate','last_active_block':'last_query'}.get(method,method)
        if model=='qwen3b':result=request3(runtime,row,mapped,protocol,controller,encoder)
        else:
            result=request7(runtime,row,mapped,runtime_cfg,controller,encoder,protocol['interaction_threshold'])
            if method=='active_block':temps=lock['frozen_temperatures']['active_block']
            elif method=='last_active_block':temps=lock['frozen_temperatures']['last_active_block']
            else:temps=[x['temperature'] for x in protocol['temperatures'][method]]
            result=attach_probability(result,temps)
        synchronize();elapsed=(time.perf_counter()-started)*1000
        result.update(method=method,raw_generation_latency_ms=result['latency_ms'],latency_ms=elapsed,
                      ttft_first_computed_ms=result['ttft_ms'],idle_allocated_by_gpu=idle,
                      peak_allocated_bytes_by_gpu=[torch.cuda.max_memory_allocated(i) for i in range(torch.cuda.device_count())])
        result['ttft_returnable_ms']=elapsed if result.get('refusal_policy')=='post_generation_replace' else result['ttft_ms']
        c=result.get('control')
        if c:
            expected=row['expected_original_route'];observed=original_route(result,method,active)
            assert observed==expected,(row['id'],expected,observed)
            result['original_gate_route']=observed
            if method in ['active_block','last_active_block']:
                assert c['route']==(1 if expected in [1,3] else expected)
                assert result.get('refusal_policy') is None
            expected_block=c['route']==1
            assert c['gate_calls']==1 and c['middle_calls']==result['generated_tokens']
            assert c['attention_calls']==(len(runtime.model.model.layers)-layer)*result['generated_tokens']*expected_block
            assert c['residual_edits']==0
            assert c.get('extra_model_calls',0)==0
        if method!='interaction':assert result['forward_calls']==result['generated_tokens']*(2 if method=='cad' else 1)
        assert np.isfinite([result['first_token']['confidence'],result['answer_confidence'],elapsed]).all()
        static=(memory['encoder_static'] if method=='interaction' else memory['active_gate_static'] if method in ['active_block','last_active_block']
                else memory['original_gate_static'] if method in ['early_gate','last_query'] else [0]*len(idle))
        transient=[max(0,int(p-i)) for p,i in zip(result['peak_allocated_bytes_by_gpu'],idle)]
        result['memory_attribution']={'static_component_bytes':static,'transient_above_shared_idle_bytes':transient,
            'increment_over_model_only_bytes':[s+t for s,t in zip(static,transient)],
            'attributed_peak_bytes':[b+s+t for b,s,t in zip(model_only,static,transient)]}
        result['timing_scope']='synchronized full method request including frozen calibration/policy; hook registration/removal excluded; no inactive SLEA hooks on baselines'
        if controller is not None:
            controller.close();live_controls.remove(controller)
        return result

    generated=0
    try:
        if model=='qwen3b':
            quality=read_jsonl(inp/'quality_requests.jsonl');assert len(quality)==234
            for i,row in enumerate(quality):
                method='active_block' if row['attention_scope']=='all_queries' else 'last_active_block'
                item=request(row,method)
                assert item['control']['route']==1 and item['original_gate_route'] in [1,3]
                item.update(closure_group=row['closure_group'],quality_cache_completion=True)
                append_jsonl(out/'quality_new.jsonl',item);generated+=1
                if generated%20==0 or generated==234:print({'stage':'quality_cache_completion','completed':generated,'total':234},flush=True)
        selected=read_jsonl(inp/'fixed_inputs.jsonl');assert len(selected)==8
        assert sum(r['state'] in ['param','abstain'] for r in selected)==4
        assert all((r['expected_original_route'] in [1,3])==(r['state'] in ['param','abstain']) for r in selected)
        runtime.fixed_length_benchmark=True
        # Warm every method on one trusted and one conflict locked input.
        warm=[next(r for r in selected if r['state']=='ext'),next(r for r in selected if r['state']=='param')]
        for row in warm:
            for method in cfg['methods']:
                item=request(row,method);assert item['generated_tokens']==12
                item.update(fixed_budget=True,warmup=True);append_jsonl(out/'warmup.jsonl',item)
        fixed=[]
        for repeat in range(3):
            for index,row in enumerate(selected):
                for order,method in enumerate(timing_order(cfg['methods'],repeat,index)):
                    item=request(row,method);assert item['generated_tokens']==12
                    item.update(repeat=repeat,input_index=index,method_order=order,fixed_budget=True,warmup=False)
                    fixed.append(item);append_jsonl(out/'fixed_workload.jsonl',item)
            print({'stage':'matched_timing','repeat_completed':repeat+1,'requests':len(fixed)},flush=True)
        assert len(fixed)==168
        for method in ['active_block','last_active_block']:
            rr=[r for r in fixed if r['method']==method]
            assert len(rr)==24 and sum(r['control']['route']==1 for r in rr)==12
            assert all((r['control']['route']==1)==(r['state'] in ['param','abstain']) for r in rr)
        dump(out/'completion.json',{'complete':True,'model':model,'quality_requests':generated,'timing_requests':168,'warmup_requests':14,
            'fixed_input_ids':[r['id'] for r in selected],'fixed_tokens':12,'repeats':3,'new_fits':0,'stage2_used':False,
            'active_hard_requests_per_active_method':12,'old_gate_hard_requests':sum(r['method']=='early_gate' and r['control']['route']==1 for r in fixed),
            'extra_arbitration_model_calls':0,'prefix_replays':0,'completed_utc':datetime.now(timezone.utc).isoformat()})
    finally:
        for controller in list(live_controls):controller.close()
        runtime.fixed_length_benchmark=False
    print('Active closure complete',model,flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',choices=['qwen3b','qwen7b'],required=True)
    p.add_argument('--check-inputs-only',action='store_true');a=p.parse_args()
    if a.check_inputs_only:print(check_inputs(a.model),flush=True)
    else:main(a.model)
