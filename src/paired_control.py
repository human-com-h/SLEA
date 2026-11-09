"""Calibration-paired correction and Oracle-only causal control diagnostics."""
import numpy as np
import torch
from .slea_controller import SLEAController


def fit_paired(cb,wrong,records,ranks=(1,4,8)):
    if any(r['split']!='calibration' or not r['prior_correct'] for r in records):
        raise ValueError('Paired correction may fit prior-correct calibration facts only')
    if len(records)!=len(cb) or np.shape(cb)!=np.shape(wrong):
        raise ValueError('Matched one-to-one paired features required')
    if len({r['fact_id'] for r in records})!=len(records) or len(records)<2:
        raise ValueError('Need unique paired facts')
    cb,wrong=np.asarray(cb,dtype=np.float64),np.asarray(wrong,dtype=np.float64)
    if cb.ndim!=2 or not np.isfinite(cb).all() or not np.isfinite(wrong).all():
        raise ValueError('Finite feature matrices required')
    # Uncentered SVD captures the mean contamination as well as its variation.
    contamination=wrong-cb
    _,singular,vt=np.linalg.svd(contamination,full_matrices=False)
    tolerance=max(contamination.shape)*np.finfo(np.float64).eps*singular[0]
    available=int(np.sum(singular>tolerance))
    basis={int(r):vt[:int(r)].T.copy() for r in ranks if int(r)<=available}
    return {'mean_delta':(cb-wrong).mean(0),'cb_center':cb.mean(0),'basis':basis,
        'singular_values':singular,'effective_rank':available,'pair_n':len(records),
        'fit_fact_ids':[r['fact_id'] for r in records]}


class PairedOracleController(SLEAController):
    """No true labels are available in deployment; this class is Oracle diagnosis."""
    def __init__(self,model,layer,paired,kind='none',strength=0.,rank=0,span_mode='none',
                 oracle_route=1,classification=None,attention_scope='last_query'):
        if kind not in ['none','classification','mean_shift','subspace']:
            raise ValueError('Unknown correction kind')
        if span_mode not in ['none','3','6','10','100','hard']:
            raise ValueError('Unknown span mode')
        if oracle_route not in range(4):raise ValueError('Explicit Oracle route required')
        if attention_scope not in ['last_query','all_queries']:raise ValueError('Unknown query scope')
        self.attention_scope=attention_scope
        self.kind,self.strength,self.rank=kind,float(strength),int(rank)
        self.span_mode=span_mode
        device=next(model.parameters()).device
        tensor=lambda value:torch.as_tensor(value,dtype=torch.float16,device=device)
        self.delta=tensor(paired['mean_delta']);self.cb_center=tensor(paired['cb_center'])
        self.basis=tensor(paired['basis'][rank]) if kind=='subspace' else None
        if kind=='classification' and classification is None:raise ValueError('Classification direction required')
        super().__init__(model,classification,layer,channel1=kind!='none' and strength!=0,
            channel2=span_mode!='none',residual_strength=strength,
            span_bias=0. if span_mode in ['none','hard'] else float(span_mode),force_route=oracle_route)

    def _middle(self,module,args,output):
        if self.span is None:return output
        h=output[0]
        if h.shape[0]!=1:raise ValueError('Oracle diagnosis supports batch=1')
        self.middle_calls+=1;self.query_lengths.append(h.shape[1])
        if self.route is None:
            self.route=torch.full((1,),self.force_route,device=h.device,dtype=torch.long)
            self.probabilities=torch.nn.functional.one_hot(self.route,4).to(torch.float16)
            self.gate_calls+=1
        # Oracle protects only param; ext/align are deliberately identity controls.
        if self.force_route!=1 or not self.channel1:return output
        last=h[:,-1,:]
        if self.kind=='mean_shift':edit=self.strength*self.delta[None,:]
        elif self.kind=='subspace':
            edit=-self.strength*((last-self.cb_center)@self.basis)@self.basis.T
        else:
            u=self.arbitrator.unit
            current=(last-self.arbitrator.center)@u
            edit=self.strength*(self.arbitrator.targets[1]-current)[:,None]*u
        changed=h.clone();changed[:,-1,:]=last+edit
        self.residual_edits+=1
        return (changed,)+output[1:]

    def _attention(self,module,args,kwargs):
        if self.span is None or self.route is None or self.force_route!=1 or self.span_mode=='none':
            return args,kwargs
        mask=kwargs.get('attention_mask')
        if mask is None or mask.ndim!=4 or mask.dtype!=torch.float16:
            raise ValueError('Expected explicit float16 eager causal mask')
        start,end=self.span
        if not 0<=start<end<=mask.shape[-1]:raise ValueError('Invalid context span')
        changed=mask.clone()
        queries=slice(None) if self.attention_scope=='all_queries' else slice(-1,None)
        if self.span_mode=='hard':changed[:,:,queries,start:end]=-float('inf')
        else:changed[:,:,queries,start:end]-=float(self.span_mode)
        kwargs=dict(kwargs);kwargs['attention_mask']=changed
        self.attention_calls+=1
        return args,kwargs

    def audit(self):
        result=super().audit()
        result.update(correction_kind=self.kind,correction_rank=self.rank,span_mode=self.span_mode,
            attention_scope=self.attention_scope,
            oracle=True,active_only_on_param=True,
            residual_scope='last prompt token and subsequent generated tokens',
            hard_mask_scope=f'upper-layer {self.attention_scope} to context keys; earlier residual/KV paths remain')
        return result
