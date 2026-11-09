from __future__ import annotations
import numpy as np
from sklearn.metrics import f1_score
from .common import exact_match,is_refusal

def binary_calibration(correct,confidence,bins=10):
    y=np.asarray(correct,dtype=float)
    p=np.asarray(confidence,dtype=float)
    if not len(y):return {'ece':None,'brier':None}
    ece=0.
    for i in range(bins):
        mask=(p>=i/bins)&((p<(i+1)/bins) if i<bins-1 else (p<=1))
        if mask.any():ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    order=np.argsort(-p,kind='stable')
    risk=1-np.cumsum(y[order])/np.arange(1,len(y)+1)
    return {'ece':float(ece),'brier':float(np.mean((p-y)**2)),
            'risk_coverage':{'coverage':(np.arange(1,len(y)+1)/len(y)).tolist(),'risk':risk.tolist()}}

def summarize(rows):
    values={}
    for state in ['ext','param','align','abstain']:
        selected=[r for r in rows if r['state']==state]
        values[state+'_n']=len(selected)
        values[state+'_accuracy']=float(np.mean([exact_match(r['answer'],r['gold_answers']) for r in selected])) if selected else None
    a,b=values['ext_accuracy'],values['param_accuracy']
    values['h_score']=2*a*b/(a+b) if a is not None and b is not None and a+b else (0. if a==0 or b==0 else None)
    values['refusal_f1']=float(f1_score([r['state']=='abstain' for r in rows],[is_refusal(r['answer']) for r in rows],zero_division=0))
    ms=np.array([r['latency_ms'] for r in rows])
    values.update(latency_mean_ms=float(ms.mean()),latency_std_ms=float(ms.std()),
                  latency_median_ms=float(np.median(ms)),ttft_mean_ms=float(np.mean([r['ttft_ms'] for r in rows])),
                  mean_generated_tokens=float(np.mean([r['generated_tokens'] for r in rows])),
                  mean_forward_calls=float(np.mean([r['forward_calls'] for r in rows])),
                  peak_allocated_bytes=max(r['peak_allocated_bytes'] for r in rows))
    ordinary=[r for r in rows if r.get('refusal_policy')!='post_generation_replace']
    # Token probability is a proxy, never label it a calibrated answer correctness probability.
    values['answer_token_proxy_calibration']=binary_calibration(
        [exact_match(r['answer'],r['gold_answers']) for r in ordinary],
        [r['answer_confidence'] for r in ordinary])
    return values
