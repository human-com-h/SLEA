"""Calibration-only factor fitting, full-dimensional pair controls and OOF screen."""
import numpy as np
import warnings
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score
from sklearn.exceptions import ConvergenceWarning


def lw_dual_weight(x, y):
    """Exact Ledoit-Wolf coefficient and Woodbury solve without a dxd matrix."""
    means = np.stack([x[y == k].mean(0) for k in [0,1]])
    r = x-means[y]; n,d = r.shape
    row_norm2 = np.sum(r*r,1); trace = row_norm2.sum()/n; mu = trace/d
    gram = r@r.T
    delta_raw = np.sum(gram*gram)/n**2
    beta = (np.sum(row_norm2**2)/n-delta_raw)/(d*n)
    delta = (delta_raw-d*mu**2)/d
    shrink = 0. if beta == 0 else min(beta,delta)/delta
    shrink = float(np.clip(shrink,1e-8,1.))
    ridge = max(shrink*mu,1e-10)
    u = r*np.sqrt((1-shrink)/n)
    direction = means[1]-means[0]
    weight = direction/ridge-u.T@np.linalg.solve(np.eye(n)+u@u.T/ridge,u@direction)/ridge**2
    bias = -.5*(means[1]+means[0])@weight
    return weight, float(bias), shrink


def prepare(x, rows, role, mode):
    x = x.astype(np.float64)
    center=x.mean(0); scale=np.maximum(x.std(0),1e-4); z=(x-center)/scale
    if role == 'known':
        yy=np.array([r['prior_correct'] for r in rows],int)
        return z,yy,center,scale
    grouped={}
    for i,r in enumerate(rows):
        if mode == 'known_pairs' and not r['prior_correct']:
            continue
        grouped.setdefault(r['fact_id'],{})[r['state'] in ['param','abstain']]=i
    delta=np.array([z[g[True]]-z[g[False]] for _,g in sorted(grouped.items())])
    return np.r_[delta,-delta],np.r_[np.ones(len(delta),int),np.zeros(len(delta),int)],center,scale


def fit_linear(x, rows, role, mode, estimator, C=None):
    z,y,center,scale=prepare(x,rows,role,mode)
    note={}
    if estimator == 'logistic':
        model=LogisticRegression(C=C,max_iter=1500,tol=1e-7,solver='lbfgs',fit_intercept=role=='known',
            class_weight='balanced',random_state=42)
        with warnings.catch_warnings():
            warnings.simplefilter('error',ConvergenceWarning); model.fit(z,y)
        w,b=model.coef_[0],float(model.intercept_[0]) if role=='known' else 0.
    elif estimator == 'lw_lda':
        w,b,shrink=lw_dual_weight(z,y);note['shrinkage']=shrink
    else:
        positive=z[y==1].mean(0);negative=z[y==0].mean(0)
        w=positive-negative;w/=max(np.linalg.norm(w),1e-12)
        b=-.5*(positive+negative)@w if role=='known' else 0.
    weight=w/scale;bias=b-center@weight
    if not np.isfinite(weight.astype(np.float16)).all() or not np.isfinite(np.float16(bias)):
        raise FloatingPointError('Folded factor not representable in float16')
    return dict(weight=weight.astype(np.float16),bias=np.float16(bias),estimator=estimator,C=C,**note)


def score(model,x):
    # CPU fp32 accumulation of fp16 operands; final score is rounded to fp16.
    return (x.astype(np.float16).astype(np.float32)@model['weight'].astype(np.float32)+np.float32(model['bias'])).astype(np.float16).astype(float)


def choose_factor(x,rows,role,mode,Cs):
    groups=[r['fact_id'] for r in rows]
    target=np.array([r['prior_correct'] if role=='known' else r['state'] in ['param','abstain'] for r in rows],int)
    known=np.array([r['prior_correct'] for r in rows],bool)
    evaluate=np.ones(len(rows),bool) if role=='known' else known
    options=[('logistic',C) for C in Cs]+[('lw_lda',None),('projection',None)]
    trials=[];best=None
    for estimator,C in options:
        values=np.zeros(len(rows));failed=None
        try:
            for train,test in GroupKFold(5).split(rows,groups=groups):
                m=fit_linear(x[train],[rows[i] for i in train],role,mode,estimator,C)
                values[test]=score(m,x[test])
            auc=float(roc_auc_score(target[evaluate],values[evaluate]))
            key=(auc,-options.index((estimator,C)))
            record=dict(estimator=estimator,C=C,oof_auc=auc,status='PASS')
            if best is None or key>best[0]:best=(key,record,values)
        except (FloatingPointError,ConvergenceWarning,np.linalg.LinAlgError) as exc:
            record=dict(estimator=estimator,C=C,status='FAILED_NOT_SELECTED',reason=str(exc))
        trials.append(record)
    assert best is not None
    chosen=best[1]; fitted=fit_linear(x,rows,role,mode,chosen['estimator'],chosen['C'])
    return fitted,best[2],dict(selected=chosen,candidates=trials)


def threshold_pair(known,conflict,rows):
    ext=np.array([r['state']=='ext' for r in rows]);param=np.array([r['state']=='param' for r in rows])
    trusted=np.array([r['state'] in ['ext','align'] for r in rows])
    quantiles=[0,.25,.5,.75,.9,.95,.975,1.]
    ks=[None]+sorted(set(float(np.float16(v)) for v in np.quantile(known,quantiles)))
    cs=[None]+sorted(set(float(np.float16(v)) for v in np.quantile(conflict,quantiles)))
    best=None;trials=[]
    for a in ks:
        for b in cs:
            mask=np.zeros(len(rows),bool) if a is None or b is None else (known>=a)&(conflict>=b)
            fpr=float(mask[ext].mean()); recall=float(mask[param].mean())
            row=dict(known_threshold=a,conflict_threshold=b,param_recall=recall,ext_false_block_rate=fpr,
                trusted_false_blocks=int(mask[trusted].sum()),total_blocks=int(mask.sum()),eligible=bool(fpr<=.01+1e-12))
            key=(recall,-row['trusted_false_blocks'],-row['total_blocks'])
            trials.append(row)
            if row['eligible'] and (best is None or key>best[0]):best=(key,row)
    return best[1],trials


def decide(k,c,policy):
    a,b=policy['known_threshold'],policy['conflict_threshold']
    return np.zeros(len(k),bool) if a is None or b is None else (k>=np.float16(a))&(c>=np.float16(b))


def select_layer(features,rows,Cs,groups):
    result=[]
    for group in groups:
        x=features[group]
        k,ko,kinfo=choose_factor(x,rows,'known','known_pairs',Cs)
        for mode in ['known_pairs','all_pairs_control']:
            c,co,cinfo=choose_factor(x,rows,'conflict',mode,Cs)
            policy,trials=threshold_pair(ko,co,rows)
            item=dict(feature_group=group,training_mode=mode,known=k,conflict=c,policy=policy,
                calibration_oof=dict(known=kinfo,conflict=cinfo,threshold_trials=trials),
                selection_key=[policy['param_recall'],-policy['trusted_false_blocks'],-policy['total_blocks'],
                    .5*(kinfo['selected']['oof_auc']+cinfo['selected']['oof_auc'])])
            result.append(item)
    chosen={mode:max([r for r in result if r['training_mode']==mode],key=lambda r:tuple(r['selection_key']))
            for mode in ['known_pairs','all_pairs_control']}
    return chosen,result
