"""Validation-only temperature calibration of answer utility, separate from route labels."""
import numpy as np
from scipy.special import expit,logit
from .common import exact_match,is_refusal
from .metrics import binary_calibration

def utility_correct(row):
    return is_refusal(row['answer']) if row['state']=='abstain' else exact_match(row['answer'],row['gold_answers'])

def raw_confidence(row):
    if row.get('refusal_policy')=='post_generation_replace':
        if row.get('control',{}).get('route_probabilities') is not None:
            return float(row['control']['route_probabilities'][3])
        return 1-float(row['answer_confidence'])
    return float(row['answer_confidence'])

def fit_temperature(rows):
    if any(row['split']!='val' for row in rows):raise ValueError('Answer calibration fits validation only')
    y=np.array([utility_correct(r) for r in rows],dtype=float)
    x=logit(np.clip([raw_confidence(r) for r in rows],1e-6,1-1e-6))
    candidates=[]
    for temperature in [.25,.5,1.,2.,4.,8.]:
        p=expit(x/temperature).clip(1e-10,1-1e-10)
        nll=-np.mean(y*np.log(p)+(1-y)*np.log1p(-p))
        candidates.append({'temperature':temperature,'nll':float(nll)})
    chosen=min(candidates,key=lambda c:c['nll'])
    return {'temperature':chosen['temperature'],'validation_n':len(rows),'candidates':candidates,
            'event':'factually correct on ext/param/align; refusal on abstain',
            'input':'answer token probability proxy; forced refusals use route abstain posterior or 1-proxy',
            'fit_split':'val'}

def evaluate_calibration(rows,spec):
    y=[utility_correct(r) for r in rows]
    p=expit(logit(np.clip([raw_confidence(r) for r in rows],1e-6,1-1e-6))/spec['temperature'])
    summary=binary_calibration(y,p)
    summary['event']=spec['event']
    summary['temperature']=spec['temperature']
    accepted=[i for i,r in enumerate(rows) if not is_refusal(r['answer'])]
    order=sorted(accepted,key=lambda i:(-p[i],i))
    correct=np.array([exact_match(rows[i]['answer'],rows[i]['gold_answers']) for i in order])
    summary['selective_answer_risk_coverage']={
        'coverage':(np.arange(1,len(order)+1)/len(rows)).tolist(),
        'risk':(1-np.cumsum(correct)/np.arange(1,len(order)+1)).tolist(),
        'denominator':len(rows),'refusals_excluded_from_answered_risk':True}
    return summary
