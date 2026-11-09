"""Development-only extraction, calibration-only screen and actual validation."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import torch
from threadpoolctl import threadpool_limits
from sklearn.metrics import roc_auc_score
from .common import ROOT, dump, read_jsonl, write_jsonl, seed_all, STATES
from .nqswap_runtime_v2 import runtime_for, environment
from .phase4_runtime import ScopeController, synchronize
from .causalgate_features_v1 import CausalFeatures, GROUPS
from .causalgate_learning_v1 import select_layer, decide
from .confirmation_statistics import bootstrap_counts, holm
from .calibration import utility_correct


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def jsonable(x):
    if isinstance(x,dict):return {k:jsonable(v) for k,v in x.items() if k not in ['weight','bias']}
    if isinstance(x,(list,tuple)):return [jsonable(v) for v in x]
    if isinstance(x,np.generic):return x.item()
    return x


def quality(rows):
    rates=np.array([np.mean([utility_correct(r) for r in rows if r['state']==s]) for s in STATES])
    return dict(rates=dict(zip(STATES,rates.tolist())),param_accuracy=float(rates[1]),
        h_score=float(2*rates[0]*rates[1]/(rates[0]+rates[1])) if rates[0]+rates[1] else 0.,macro_utility=float(rates.mean()))


def compose(base,hard,mask):
    assert [r['id'] for r in base]==[r['id'] for r in hard]
    return [dict(h if m else b,intercepted=bool(m),cache_composed=True) for b,h,m in zip(base,hard,mask)]


def packed(candidate,layer):
    return dict(weights=np.column_stack([candidate['known']['weight'],candidate['conflict']['weight']]).astype(np.float16),
        bias=np.array([candidate['known']['bias'],candidate['conflict']['bias']],np.float16),layer=layer)


def gpu_scores(x,saved,device):
    w=torch.tensor(saved['weights'],device=device,dtype=torch.float16);b=torch.tensor(saved['bias'],device=device,dtype=torch.float16)
    with torch.inference_mode():
        return np.stack([torch.addmm(b,torch.tensor(row[None],device=device,dtype=torch.float16),w)[0].float().cpu().numpy() for row in x])


class FactorGate:
    def __init__(self,collector,layer,group,saved,policy,device):
        self.collector,self.layer,self.group,self.policy=collector,layer,group,policy
        self.weight=torch.tensor(saved['weights'],device=device,dtype=torch.float16)
        self.bias=torch.tensor(saved['bias'],device=device,dtype=torch.float16)
    def decide(self,values):
        x=self.collector.vector(self.layer,self.group).to(self.weight.device)[None]
        s=torch.addmm(self.bias,x,self.weight)
        a,b=self.policy['known_threshold'],self.policy['conflict_threshold']
        active=torch.zeros(1,device=s.device,dtype=torch.bool) if a is None or b is None else (s[:,0]>=a)&(s[:,1]>=b)
        route=active.to(torch.long);action=torch.nn.functional.one_hot(route,4).to(torch.float16)
        return route,action,s


class CausalController(ScopeController):
    def __init__(self,runtime,collector,layer,gate):
        self.collector=collector
        super().__init__(runtime,layer,gate=gate,scope='all_queries')
    def reset(self,span):
        super().reset(span);self.collector.reset(span)
    def audit(self):
        r=super().audit();r['two_factor_scores']=r.pop('folded_four_logits');r['route_probabilities_semantics']='one-hot action indicator, not epistemic posterior'
        r['causal_feature_audit']=self.collector.audit();return r


def paired_inference(base,rows):
    facts=sorted({r['fact_id'] for r in base});ix={f:i for i,f in enumerate(facts)}
    def matrix(rr):
        a=np.zeros((len(facts),4));den=np.zeros_like(a);known=np.zeros(len(facts),bool)
        for r in rr:
            i,j=ix[r['fact_id']],STATES.index(r['state']);a[i,j]=utility_correct(r);den[i,j]=1;known[i]=r['prior_correct']
        return a,den.sum(0),known
    b,den,known=matrix(base);a,_,_=matrix(rows)
    counts=bootstrap_counts(known,10000,42);swap=np.random.default_rng(42).integers(0,2,size=(10000,len(facts)))
    def metric(z):
        r=np.atleast_2d(z)/den;e,p=r[:,0],r[:,1]
        return np.column_stack([p,np.divide(2*e*p,e+p,out=np.zeros_like(p),where=e+p!=0),r.mean(1)])
    sa,sb=a.sum(0),b.sum(0);delta=(metric(sa)-metric(sb))[0]
    ci=np.quantile(metric(counts@a)-metric(counts@b),[.025,.975],axis=0)
    mixed=sb+swap@(a-b);null=metric(mixed)-metric(sa+sb-mixed)
    pp=(1+np.sum(abs(null)>=abs(delta)[None]-1e-14,axis=0))/10001
    return {m:dict(delta=float(delta[j]),ci95=ci[:,j].tolist(),p=float(pp[j])) for j,m in enumerate(['param_accuracy','h_score','macro_utility'])}


def run(model):
    assert Path('/kaggle/working').is_dir()
    cfg=json.loads((ROOT/'configs/nqswap_causalgate_stage_a_v1.json').read_text())
    inputs=ROOT/'data/processed/causalgate_stage_a_v1'/model
    lock=json.loads((inputs/'input_lock.json').read_text())
    for name,digest in lock['files'].items():assert sha(inputs/name)==digest,name
    assert not list((ROOT/'data').rglob('*blind*')) and not list((ROOT/'data').rglob('*holdout*'))
    out=ROOT/'results/causalgate_stage_a_v1'/model;out.mkdir(parents=True,exist_ok=True)
    assert not (out/'entry.json').exists()
    dump(out/'entry.json',dict(seed_state=seed_all(42),input_lock=lock,stage_b_access=False,old_blind_access=False))
    rows=read_jsonl(inputs/'development.jsonl');assert len(rows)==800
    cal=[r for r in rows if r['split']=='calibration'];val=[r for r in rows if r['split']=='val']
    assert len(cal)==500 and len(val)==300
    runtime=runtime_for(model,32);layers=cfg['candidate_layers'][model]
    dump(out/'environment.json',environment(runtime,model))
    capture=CausalFeatures(runtime.model,layers);arrays={};audits=[]
    try:
        for i,r in enumerate(rows):
            encoded,meta=runtime.encode(r['question'],r['context']);capture.reset(meta['context_span'])
            before=runtime.calls
            with torch.inference_mode():runtime.model(**encoded,use_cache=False,num_logits_to_keep=1)
            synchronize();assert runtime.calls-before==1
            for l in layers:
                for k,v in capture.values[l].items():arrays.setdefault(f'l{l}_{k}',[]).append(v.cpu().numpy())
                arrays.setdefault(f'l{l}_raw_last',[]).append(capture.raw[l]['last'].cpu().numpy())
            audits.append(dict(id=r['id'],forward_calls=1,**capture.audit()))
            if (i+1)%50==0:print(dict(stage='causal_features',model=model,completed=i+1,total=800),flush=True)
    finally:capture.close()
    arrays={k:np.stack(v) for k,v in arrays.items()};np.savez_compressed(out/'causal_features.npz',ids=[r['id'] for r in rows],**arrays)
    write_jsonl(out/'causal_extraction_audit.jsonl',audits)
    calmask=np.array([r['split']=='calibration' for r in rows]);valmask=~calmask
    original_layer=10 if model=='qwen3b' else 11
    with np.load(inputs/'old_development_features.npz') as old:
        maximum=max(float(np.max(np.abs(arrays[f'l{original_layer}_raw_last'][mask].astype(float)-old[key].astype(float))))
                    for mask,key in [(calmask,'calibration'),(valmask,'validation')])
    dump(out/'feature_cache_equivalence.json',dict(original_layer=original_layer,max_abs_difference=maximum,
        note='same development inputs only; no old blind features loaded'))
    assert maximum<=.01,'Early features differ from the original development capture; investigate before fitting'
    baseline=read_jsonl(inputs/'validation_vanilla.jsonl');hard={}
    for l in layers:
        existing=inputs/f'validation_hard_l{l}.jsonl'
        if existing.is_file():hard[l]=read_jsonl(existing)
        else:
            controller=ScopeController(runtime,l,force_route=1,scope='all_queries');rr=[]
            try:
                for i,r in enumerate(val):
                    result=runtime.generate(r['question'],r['context'],controller=controller)
                    n=result['generated_tokens'];control=controller.audit()
                    assert result['forward_calls']==n and control['attention_calls']==(len(runtime.model.model.layers)-l)*n
                    rr.append(dict(r,**result,control=control))
                    if (i+1)%50==0:print(dict(stage='validation_hard',layer=l,completed=i+1,total=300),flush=True)
            finally:controller.close()
            hard[l]=rr;write_jsonl(out/f'validation_hard_l{l}.jsonl',rr)
    candidates=[]
    for l in layers:
        features={g:np.concatenate([arrays[f'l{l}_{k}'] for k in GROUPS[g]],axis=1) for g in cfg['feature_groups']}
        chosen,all_trials=select_layer({g:x[calmask] for g,x in features.items()},cal,cfg['C'],cfg['feature_groups'])
        dump(out/f'calibration_screen_l{l}.json',jsonable(all_trials))
        for mode,candidate in chosen.items():
            saved=packed(candidate,l);device=next(runtime.model.model.layers[l-1].parameters()).device
            scores=gpu_scores(features[candidate['feature_group']][valmask],saved,device)
            mask=decide(scores[:,0],scores[:,1],candidate['policy']);composed=compose(baseline,hard[l],mask)
            q=quality(composed);old=quality(baseline)
            ext=np.array([r['state']=='ext' for r in val]);unknown=np.array([not r['prior_correct'] for r in val])
            accepted_guard=bool(q['rates']['ext']>=old['rates']['ext']-.01-1e-12 and mask[ext].mean()<=.01+1e-12)
            known_y=np.array([r['prior_correct'] for r in val]);conf_y=np.array([r['state']=='param' for r in val])
            record=dict(layer=l,training_mode=mode,feature_group=candidate['feature_group'],policy=candidate['policy'],
                quality=q,oracle_hard_quality=quality(hard[l]),baseline_quality=old,
                calibration_oof=jsonable(candidate['calibration_oof']),triggered=int(mask.sum()),ext_false_blocks=int(mask[ext].sum()),
                ext_false_block_rate=float(mask[ext].mean()),ext_guard=accepted_guard,
                validation_known_auc=float(roc_auc_score(known_y,scores[:,0])),
                validation_known_conflict_auc=float(roc_auc_score(conf_y[known_y],scores[:,1][known_y])),
                inference=paired_inference(baseline,composed),factor_fits=jsonable(dict(known=candidate['known'],conflict=candidate['conflict'])))
            key=f'l{l}_{mode}';np.savez(out/(key+'_gate.npz'),**saved)
            record['weights_file']=key+'_gate.npz';record['weights_sha256']=sha(out/record['weights_file'])
            write_jsonl(out/(key+'_cache_validation.jsonl'),composed)
            candidates.append(record)
        print(dict(stage='calibration_screen',layer=l,complete=True),flush=True)
    main=[c for c in candidates if c['training_mode']=='known_pairs']
    pool=[(c,m) for c in main for m in ['param_accuracy','h_score','macro_utility']]
    for (c,m),p in zip(pool,holm([c['inference'][m]['p'] for c,m in pool])):c['inference'][m]['p_holm12']=p
    for c in main:
        c['statistical_gain_pass']=bool(all(c['inference'][m]['delta']>0 and c['inference'][m]['p_holm12']<.05 for m in ['param_accuracy','h_score','macro_utility']))
        c['provisional_pass']=c['ext_guard'] and c['statistical_gain_pass']
    # Always audit the strongest guard-satisfying candidate, even if gains did not pass.
    eligible=[c for c in main if c['ext_guard']] or main
    selected=sorted(eligible,key=lambda c:(-int(c['provisional_pass']),-c['quality']['macro_utility'],-c['quality']['h_score'],c['layer']))[0]
    dump(out/'validation_selection.json',dict(candidates=candidates,selected=selected,stage_b_frozen=False))
    l=selected['layer'];collector=CausalFeatures(runtime.model,[l],selected['feature_group'])
    with np.load(out/selected['weights_file']) as z:saved={k:z[k] for k in z.files}
    gate=FactorGate(collector,l,selected['feature_group'],saved,selected['policy'],next(runtime.model.model.layers[l-1].parameters()).device)
    controller=CausalController(runtime,collector,l,gate);actual=[];mismatches=[]
    expected=read_jsonl(out/f'l{l}_known_pairs_cache_validation.jsonl')
    try:
        for i,(r,e) in enumerate(zip(val,expected)):
            result=runtime.generate(r['question'],r['context'],controller=controller);synchronize();control=controller.audit()
            n=result['generated_tokens'];assert result['forward_calls']==n and control['gate_calls']==1
            assert control['attention_calls']==(len(runtime.model.model.layers)-l)*n*control['block']
            item=dict(r,**result,control=control);actual.append(item)
            if result['answer']!=e['answer'] or control['block']!=e['intercepted']:mismatches.append(r['id'])
            if (i+1)%50==0:print(dict(stage='actual_causal_validation',completed=i+1,total=300),flush=True)
    finally:controller.close();collector.close()
    write_jsonl(out/'actual_causal_validation.jsonl',actual)
    passed=bool(selected['provisional_pass'] and not mismatches)
    dump(out/'completion.json',dict(stage='A',model=model,accepted=passed,selected=selected,actual_quality=quality(actual),
        actual_requests=300,actual_hard_requests=sum(r['control']['block'] for r in actual),mismatched_ids=mismatches,
        total_backbone_calls=runtime.calls,feature_prefills=800,stage_b_access=False,old_blind_access=False,
        stage_b_freeze_created=False,user_confirmation_required=True,
        lens_timing_summary={str(l):dict(mean_ms=float(np.mean([t['ms'] for a in audits for t in a['lens_ms'] if t['layer']==l])),
            sd_ms=float(np.std([t['ms'] for a in audits for t in a['lens_ms'] if t['layer']==l]))) for l in layers}))
    print(dict(stage='A_complete',model=model,accepted=passed,stage_B_started=False),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--model',required=True,choices=['qwen3b','qwen7b'])
    with threadpool_limits(limits=4):run(p.parse_args().model)
