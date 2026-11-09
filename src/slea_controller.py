"""In-flight gate, residual direction control and upper-layer span logit bias.

The decision is computed on the first prefill only. Earlier prompt KV entries are
not recomputed; later generated query tokens reuse the same per-sample decision.
"""
from __future__ import annotations
import torch

class SLEAController:
    def __init__(self,model,arbitrator,layer,channel1=True,channel2=True,
                 residual_strength=.5,span_bias=2.,force_route=None):
        if model.config._attn_implementation!='eager':
            raise ValueError('Only verified eager attention is supported')
        if not 1<=layer<len(model.model.layers):
            raise ValueError('Gate layer must leave at least one subsequent decoder layer')
        self.model,self.arbitrator,self.layer=model,arbitrator,layer
        self.channel1,self.channel2=channel1,channel2
        self.residual_strength,self.span_bias=residual_strength,span_bias
        self.force_route=force_route
        self.handles=[model.model.layers[layer-1].register_forward_hook(self._middle)]
        if channel2:
            for decoder in model.model.layers[layer:]:
                self.handles.append(decoder.self_attn.register_forward_pre_hook(self._attention,with_kwargs=True))
        self.reset(None)

    def reset(self,span):
        self.span=span
        self.route=None
        self.probabilities=None
        self.gate_calls=0
        self.middle_calls=0
        self.attention_calls=0
        self.residual_edits=0
        self.query_lengths=[]

    def _middle(self,module,args,output):
        if self.span is None:
            return output
        h=output[0]
        if h.shape[0]!=1:
            raise ValueError('Verified implementation is batch=1 only')
        self.middle_calls+=1
        self.query_lengths.append(h.shape[1])
        last=h[:,-1,:]
        if self.route is None:
            if self.force_route is None:
                self.route,self.probabilities=self.arbitrator(last)
            else:
                self.route=torch.full((1,),self.force_route,device=h.device,dtype=torch.long)
                self.probabilities=torch.nn.functional.one_hot(self.route,4).to(torch.float16)
            self.gate_calls+=1
        if not self.channel1 or self.residual_strength==0:
            return output
        u=self.arbitrator.unit.to(h.device)
        center=self.arbitrator.center.to(h.device)
        targets=self.arbitrator.targets.to(h.device)
        target=targets[self.route]
        current=(last-center)@u
        amount=(target-current)*self.residual_strength
        amount=torch.where(self.route<2,amount,torch.zeros_like(amount))
        changed=h.clone()
        changed[:,-1,:]=last+amount[:,None]*u
        self.residual_edits+=1
        return (changed,)+output[1:]

    def _attention(self,module,args,kwargs):
        if self.span is None or self.route is None or self.span_bias==0:
            return args,kwargs
        mask=kwargs.get('attention_mask')
        if mask is None or mask.ndim!=4:
            raise ValueError('Expected explicit 4D causal additive mask with eager attention')
        if mask.dtype!=torch.float16:
            raise TypeError('Attention bias must retain float16')
        start,end=self.span
        if not 0<=start<end<=mask.shape[-1]:
            raise ValueError('Context span is outside cached key positions')
        beta=torch.where(self.route==0,self.span_bias,
                         torch.where(self.route==1,-self.span_bias,0.)).to(device=mask.device,dtype=mask.dtype)
        changed=mask.clone()
        # Only final prompt query in prefill; all decode calls have a single query.
        changed[:,:,-1,start:end]=changed[:,:,-1,start:end]+beta[:,None,None]
        kwargs=dict(kwargs)
        kwargs['attention_mask']=changed
        self.attention_calls+=1
        return args,kwargs

    def audit(self):
        return {'gate_calls':self.gate_calls,'middle_calls':self.middle_calls,
                'attention_calls':self.attention_calls,'residual_edits':self.residual_edits,
                'query_lengths':self.query_lengths,
                'route':int(self.route.item()) if self.route is not None else None,
                'route_probabilities':self.probabilities[0].cpu().tolist() if self.probabilities is not None else None,
                'gate_layer':self.layer,'channel1':self.channel1,'channel2':self.channel2,
                'residual_strength':self.residual_strength,'span_bias':self.span_bias,
                'cache_policy':'one prefill decision, reuse cached KV and route during ordinary autoregressive decoding'}

    def close(self):
        for handle in self.handles:
            handle.remove()
        self.handles=[]
