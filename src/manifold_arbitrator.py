"""Offline float64 fitting; float16 deployable direction and quadratic energy."""
from __future__ import annotations
import argparse
import math
from pathlib import Path
import numpy as np
import torch
from scipy.special import expit, softmax
from sklearn.covariance import LedoitWolf
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, log_loss, roc_auc_score
from .common import ROOT, STATES, SEEDS, dump, fingerprint, read_jsonl, seed_all

def safe_auc(y,s):
    return float(roc_auc_score(y,s)) if len(np.unique(y))==2 else None

class ManifoldArbitrator:
    def fit(self,x,y,rank=32,seed=42):
        x=np.asarray(x,dtype=np.float64)
        y=np.asarray(y)
        if any(np.sum(y==k)<2 for k in range(4)):
            raise ValueError('Need at least two calibration samples in each native state')
        rank=min(rank,len(x)-4,x.shape[1])
        pca=PCA(n_components=rank,svd_solver='randomized',random_state=seed)
        pca.fit(x)
        z=pca.transform(x)
        scale=np.sqrt(np.maximum(pca.explained_variance_,1e-10))
        z=z/scale
        self.center=pca.mean_
        self.projection=pca.components_.T/scale
        self.rank=rank
        self.means=np.stack([z[y==k].mean(0) for k in range(4)])
        self.priors=np.array([np.mean(y==k) for k in range(4)])
        self.precisions,self.logdets,self.conditions=[],[],[]
        for k in range(4):
            estimator=LedoitWolf(assume_centered=True).fit(z[y==k]-self.means[k])
            covariance=estimator.covariance_
            sign,ld=np.linalg.slogdet(covariance)
            if sign!=1:
                raise ValueError('Non-positive definite covariance')
            self.precisions.append(np.linalg.solve(covariance,np.eye(rank)))
            self.logdets.append(ld)
            self.conditions.append(float(np.linalg.cond(covariance)))
        self.precisions=np.stack(self.precisions)
        self.logdets=np.asarray(self.logdets)
        binary=(y<2)
        residual=z[binary]-self.means[y[binary]]
        pooled=LedoitWolf(assume_centered=True).fit(residual).covariance_
        w=np.linalg.solve(pooled,self.means[0]-self.means[1])
        offset=-.5*(self.means[0]+self.means[1])@w+np.log(self.priors[0]/self.priors[1])
        direction_scale=max(np.std(z[binary]@w+offset),1.)
        self.direction=self.projection@w/direction_scale
        self.offset=float(offset/direction_scale)
        # Raw-space unit direction is also the residual intervention direction.
        norm=np.linalg.norm(self.direction)
        if norm<1e-12 or not np.isfinite(norm):raise ValueError('Degenerate direction')
        self.unit=self.direction/norm
        self.targets=np.array([(x[y==k]-self.center).mean(0)@self.unit for k in range(4)])
        self.temperature=1.
        self.reject_threshold=float('inf')
        self.diagnostics={'fit_dtype':'float64','online_dtype':'float16','rank':rank,
            'calibration_counts':[int(np.sum(y==k)) for k in range(4)],
            'class_covariance_condition_numbers':self.conditions,
            'singular_value_min':float(pca.singular_values_.min()),
            'explained_variance_ratio_sum':float(pca.explained_variance_ratio_.sum()),
            'direction_score_rescaling':float(direction_scale)}
        return self

    def logits64(self,x):
        centered=np.asarray(x,dtype=np.float64)-self.center
        z=centered@self.projection
        delta=z[:,None,:]-self.means[None,:,:]
        # Per-dimension energy avoids fp16 overflow online; exact positive scaling.
        energy=np.einsum('nkr,krs,nks->nk',delta,self.precisions,delta)/self.rank
        q=-.5*(energy+self.logdets/self.rank)+np.log(self.priors)/self.rank
        direction=centered@self.direction+self.offset
        mass=np.logaddexp(q[:,0],q[:,1])
        q[:,0]=mass-np.logaddexp(0,-direction)
        q[:,1]=mass-np.logaddexp(0,direction)
        return q,energy,direction

    def predict_proba(self,x):
        logits,energy,_=self.logits64(x)
        p=softmax(logits/self.temperature,axis=-1)
        rejected=energy[:,:3].min(-1)>self.reject_threshold
        p[rejected]=np.eye(4)[3]
        return p

    def calibrate(self,x,y):
        logits,energy,_=self.logits64(x)
        # Validation only. Explicit abstain class and absolute OOD threshold are distinct.
        trusted=energy[y!=3,:3].min(-1)
        candidates=[float('inf')]
        if len(trusted):
            candidates += [float(np.quantile(trusted,q)) for q in [.90,.95,.99]]
        trials=[]
        for threshold in candidates:
            for temperature in [.25,.5,1.,2.,4.]:
                p=softmax(logits/temperature,axis=-1)
                reject=energy[:,:3].min(-1)>threshold
                p[reject]=np.eye(4)[3]
                loss=float(log_loss(y,np.clip(p,1e-10,1),labels=list(range(4))))
                trials.append((loss,temperature,threshold))
        _,self.temperature,self.reject_threshold=min(trials)
        return [{'nll':a,'temperature':b,'reject_threshold':c if np.isfinite(c) else None} for a,b,c in trials]

    def save(self,path,metadata):
        arrays={k:getattr(self,k) for k in ['center','projection','means','priors','precisions','logdets','direction','unit','targets']}
        arrays.update(rank=self.rank,offset=self.offset,temperature=self.temperature,reject_threshold=self.reject_threshold)
        np.savez(path,**arrays)
        dump(str(path)+'.json',{'metadata':metadata,'diagnostics':self.diagnostics,
             'states':STATES,'temperature':self.temperature,
             'reject_threshold':self.reject_threshold if np.isfinite(self.reject_threshold) else None,
             'method':'LDA directional ext/param mass + 4-class shrinkage QDA + absolute trusted-cluster energy rejection'})

    @classmethod
    def load(cls,path):
        obj=cls()
        with np.load(path) as a:
            for k in a.files:
                setattr(obj,k,a[k].item() if a[k].ndim==0 else a[k])
        return obj

    def deploy(self,device='cuda'):
        return OnlineArbitrator(self,device)

class OnlineArbitrator:
    def __init__(self,offline,device):
        self.rank=offline.rank
        for key in ['center','projection','means','priors','precisions','logdets','direction','unit','targets']:
            setattr(self,key,torch.as_tensor(getattr(offline,key),device=device,dtype=torch.float16))
        self.offset=torch.tensor(offline.offset,device=device,dtype=torch.float16)
        self.temperature=offline.temperature
        self.threshold=offline.reject_threshold
        self.set_prior_weight(1.)

    def set_prior_weight(self,weight):
        if weight not in [1.,2.,4.,8.,16.]:raise ValueError('Unsupported predeclared routing weight')
        self.prior_weight=float(weight)
        self.decision_bias=torch.tensor([0.,math.log(weight),math.log(weight),0.],device=self.center.device,dtype=torch.float16)

    def scores(self,h):
        if h.dtype!=torch.float16:
            raise TypeError('Online input must be float16')
        centered=h-self.center
        z=centered@self.projection
        delta=z[:,None,:]-self.means[None,:,:]
        energy=torch.einsum('nkr,krs,nks->nk',delta,self.precisions/self.rank,delta)
        q=-.5*(energy+self.logdets/self.rank)+self.priors.log()/self.rank
        direction=centered@self.direction+self.offset
        mass=torch.logaddexp(q[:,0],q[:,1])
        q[:,0]=mass-torch.nn.functional.softplus(-direction)
        q[:,1]=mass-torch.nn.functional.softplus(direction)
        return q,energy,direction

    def __call__(self,h):
        q,e,_=self.scores(h)
        p=(q/self.temperature).softmax(-1)
        route=(q/self.temperature+self.decision_bias).argmax(-1) if self.prior_weight!=1. else p.argmax(-1)
        unsafe=(e[:,:3].min(-1).values>self.threshold)|(~torch.isfinite(q).all(-1))
        route=torch.where(unsafe,torch.full_like(route,3),route)
        forced=torch.nn.functional.one_hot(torch.full_like(route,3),4).to(p.dtype)
        p=torch.where(unsafe[:,None],forced,p)
        return route,p

def scan(data,output,seed=42,anchors=500,ranks=(8,16,32,64,128)):
    seed_all(seed)
    data,output=Path(data),Path(output)
    output.mkdir(parents=True,exist_ok=True)
    rows=read_jsonl(data/'native.jsonl')
    x=np.load(data/'all_layers.npy',mmap_mode='r')
    labels=np.array([STATES.index(r['state']) for r in rows])
    cal=np.array([i for i,r in enumerate(rows) if r['split']=='calibration'])
    val=np.array([i for i,r in enumerate(rows) if r['split']=='val'])
    # Test indices are never read during selection.
    if len(cal)<anchors:
        raise ValueError(f'Only {len(cal)} calibration samples; requested {anchors}')
    rng=np.random.default_rng(seed)
    cal=rng.permutation(cal)[:anchors]
    median=float(np.median([rows[i]['prior_confidence'] for i in val]))
    records=[]
    failures=[]
    best=None
    for layer in range(x.shape[1]):
        for rank in ranks:
            try:
                model=ManifoldArbitrator().fit(x[cal,layer],labels[cal],rank,seed)
                trials=model.calibrate(x[val,layer],labels[val])
            except (ValueError,np.linalg.LinAlgError) as exc:
                failures.append({'layer':layer+1,'requested_rank':rank,'reason':str(exc)})
                dump(output/'failed_configurations.json',failures)
                continue
            probs=model.predict_proba(x[val,layer])
            binary=labels[val]<2
            direction=model.logits64(x[val,layer])[2]
            prior=np.array([rows[i]['prior_confidence'] for i in val])
            record={'layer':layer+1,'rank':model.rank,'requested_rank':rank,'seed':seed,'anchors':anchors,
                'eligible_for_upper_layer_control':layer+1<x.shape[1],
                'val_nll':float(log_loss(labels[val],probs.clip(1e-10,1),labels=list(range(4)))),
                'val_macro_f1':float(f1_score(labels[val],probs.argmax(-1),average='macro',zero_division=0)),
                'direction_auroc':safe_auc((labels[val][binary]==0).astype(int),direction[binary]),
                'direction_auroc_low_prior':safe_auc((labels[val][binary&(prior<=median)]==0).astype(int),direction[binary&(prior<=median)]),
                'direction_auroc_high_prior':safe_auc((labels[val][binary&(prior>median)]==0).astype(int),direction[binary&(prior>median)])}
            records.append(record)
            key=(record['val_nll'],-record['val_macro_f1'],layer,rank)
            if record['eligible_for_upper_layer_control'] and (best is None or key<best[0]):
                best=(key,model,record,trials)
        dump(output/'layer_scan.json',records)
        print({'layer':layer+1,'best_val_nll':best[2]['val_nll'] if best else None},flush=True)
    if best is None:raise ValueError('No usable configuration with a later layer available')
    dump(output/'failed_configurations.json',failures)
    metadata={'model_revision':json_sources()['Qwen/Qwen2.5-3B-Instruct']['revision'],
              'split_id':rows[0]['split_id'],'seed':seed,'anchor_ids':[rows[i]['id'] for i in cal],
              'validation_ids':[rows[i]['id'] for i in val],'selected':best[2],
              'selection_rule':'minimum validation multiclass NLL, then macro F1, layer, rank; final layer scanned but ineligible for upper-layer control',
              'prior_confidence_val_median':median}
    best[1].save(output/'arbitrator.npz',metadata)
    dump(output/'layer_scan.json',records)
    dump(output/'temperature_threshold_search.json',best[3])
    # Same calibration features and validation samples for lightweight geometric controls.
    layer=best[2]['layer']-1
    off=best[1]
    zcal=(x[cal,layer].astype(np.float64)-off.center)@off.projection
    zval=(x[val,layer].astype(np.float64)-off.center)@off.projection
    controls=[]
    for c in [.01,.1,1.,10.]:
        lr=LogisticRegression(C=c,max_iter=2000,random_state=seed).fit(zcal,labels[cal])
        p=lr.predict_proba(zval)
        controls.append({'method':'PCA Logistic','C':c,'val_nll':float(log_loss(labels[val],p,labels=list(range(4)))),
                         'macro_f1':float(f1_score(labels[val],p.argmax(-1),average='macro',zero_division=0))})
    delta=off.means[0]-off.means[1]
    mask=labels[val]<2
    controls.append({'method':'Mean-Diff','direction_auroc':safe_auc((labels[val][mask]==0).astype(int),(zval@delta)[mask])})
    dump(output/'geometric_controls.json',controls)
    plot_scan(records,x[val,layer],labels[val],off,output,seed)
    print(best[2],flush=True)

def json_sources():
    import json
    return json.loads((ROOT/'configs/sources.json').read_text(encoding='utf-8'))

def plot_scan(records,x,y,model,output,seed):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(7,4))
    for rank in sorted({r['rank'] for r in records}):
        r=[v for v in records if v['rank']==rank]
        ax.plot([v['layer'] for v in r],[v['direction_auroc'] or np.nan for v in r],label=f'r={rank}')
    ax.set(xlabel='Decoder layer (1-based)',ylabel='Validation directional AUROC',ylim=(0,1))
    ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(output/f'layer_auroc.{ext}',dpi=180)
    plt.close(fig)
    z=(x.astype(np.float64)-model.center)@model.projection
    fig,ax=plt.subplots(figsize=(6,5))
    for k,name in enumerate(STATES):
        ax.scatter(z[y==k,0],z[y==k,1],s=12,alpha=.6,label=name)
    ax.set(xlabel='Calibration PCA component 1 (whitened)',ylabel='Calibration PCA component 2 (whitened)',title='Independent validation features')
    ax.legend();fig.tight_layout()
    for ext in ['png','pdf']:fig.savefig(output/f'manifold.{ext}',dpi=180)
    plt.close(fig)

if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--data',default='data/processed/conflictqa_qwen25_3b_v1')
    p.add_argument('--output',default='results/qwen25_3b_seed42')
    p.add_argument('--seed',type=int,default=42,choices=SEEDS)
    p.add_argument('--anchors',type=int,default=500)
    a=p.parse_args();scan(a.data,a.output,a.seed,a.anchors)
