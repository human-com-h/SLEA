"""Early automatic hard-mask controller: one prefill decision, no label access."""
from __future__ import annotations
import time
import numpy as np
import torch
from .slea_controller import SLEAController


class FoldedGate:
    def __init__(self, file, policy, device):
        saved=np.load(file,allow_pickle=False)
        self.layers=saved['layers'].tolist()
        self.weight=torch.as_tensor(saved['weights'].reshape(-1,4),device=device,dtype=torch.float16)
        self.bias=torch.as_tensor(saved['bias'],device=device,dtype=torch.float16)
        self.cost=torch.tensor([1.,policy['param_cost'],1.,policy['abstain_cost']],device=device,dtype=torch.float16)
        self.param_threshold=policy['param_threshold']; self.abstain_threshold=policy['abstain_threshold']
        self.graph=None
        if self.weight.is_cuda:self.prepare_graph()

    @torch.inference_mode()
    def prepare_graph(self):
        # Fixed shape, gate-only CUDA graph. No transformer layer/prefix is captured
        # or replayed. This removes Python launch overhead for tiny pointwise kernels.
        self.static_input=torch.zeros((1,self.weight.shape[0]),device=self.weight.device,dtype=torch.float16)
        stream=torch.cuda.Stream(device=self.weight.device)
        stream.wait_stream(torch.cuda.current_stream(self.weight.device))
        with torch.cuda.stream(stream):
            for _ in range(5):self._decide_vector(self.static_input)
        torch.cuda.current_stream(self.weight.device).wait_stream(stream)
        self.graph=torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):self.static_output=self._decide_vector(self.static_input)

    def scores(self, values):
        vector=torch.cat(values,dim=-1) if len(values)>1 else values[0]
        return torch.addmm(self.bias,vector,self.weight)

    def decide(self, values):
        if self.graph is not None:
            if len(values)>1:torch.cat(values,dim=-1,out=self.static_input)
            else:self.static_input.copy_(values[0])
            self.graph.replay()
            return self.static_output
        vector=torch.cat(values,dim=-1) if len(values)>1 else values[0]
        return self._decide_vector(vector)

    def _decide_vector(self,vector):
        logits=torch.addmm(self.bias,vector,self.weight)
        probability=logits.softmax(-1)
        adjusted=probability*self.cost; adjusted=adjusted/adjusted.sum(-1,keepdim=True)
        winner=adjusted.argmax(-1)
        block=(adjusted[:,1]>=self.param_threshold)&(winner==1)
        refuse=(~block)&(adjusted[:,3]>=self.abstain_threshold)&(winner==3)
        route=torch.where(block,1,torch.where(refuse,3,torch.where(probability[:,2]>probability[:,0],2,0)))
        return route,probability,logits


class EarlyController(SLEAController):
    def __init__(self,model,gate):
        self.gate=gate
        super().__init__(model,gate,max(gate.layers),channel1=False,channel2=True,residual_strength=0.,span_bias=0.)
        # Parent installs the target-layer hook. Earlier hooks retain only a view of the last token.
        for layer in gate.layers[:-1]:
            self.handles.append(model.model.layers[layer-1].register_forward_hook(self._collect(layer)))

    def reset(self,span):
        super().reset(span)
        self.features={}; self.block=False; self.gate_host_ms=0.; self.logits=None

    def _collect(self,layer):
        def hook(module,args,output):
            if self.span is not None and self.route is None:
                self.features[layer]=output[0][:,-1,:].detach()
        return hook

    def _middle(self,module,args,output):
        if self.span is None:return output
        h=output[0]
        if h.shape[0]!=1:raise ValueError('Early gate supports only verified batch=1')
        self.middle_calls+=1; self.query_lengths.append(h.shape[1])
        if self.route is None:
            start=time.perf_counter()
            self.features[self.layer]=h[:,-1,:]
            self.route,self.probabilities,self.logits=self.gate.decide([self.features[l] for l in self.gate.layers])
            # One host decision; avoids a device-to-host synchronization at each upper attention layer.
            self.block=bool((self.route==1).item())
            self.gate_host_ms=(time.perf_counter()-start)*1000
            self.features.clear(); self.gate_calls+=1
        return output

    def _attention(self,module,args,kwargs):
        if self.span is None or not self.block:return args,kwargs
        mask=kwargs.get('attention_mask')
        if mask is None or mask.ndim!=4 or mask.dtype!=torch.float16:
            raise ValueError('Explicit float16 eager causal attention mask required')
        start,end=self.span
        if not 0<=start<end<=mask.shape[-1]:raise ValueError('Invalid context key span')
        changed=mask.clone(); changed[:,:,:,start:end]=-float('inf')
        kwargs=dict(kwargs);kwargs['attention_mask']=changed;self.attention_calls+=1
        return args,kwargs

    def audit(self):
        result=super().audit()
        result.update(oracle=False,feature_layers=self.gate.layers,feature_calls=1,
            gate_host_ms=self.gate_host_ms,block=self.block,attention_scope='all_queries',span_mode='hard',
            folded_four_logits=self.logits[0].float().cpu().tolist() if self.logits is not None else None,
            eta=0.,extra_model_calls=0,
            hard_mask_scope='all upper-layer prefill and decode queries to context keys; early residual/KV remain')
        return result
