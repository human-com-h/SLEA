"""Batch-one fp16 runtime, exact post-template spans, cached greedy generation."""
from __future__ import annotations
import time
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from .common import ROOT, configure_cache, cuda_sync

SYSTEM = ('Answer the question with only the shortest factual answer, without explanation. '
          'Use relevant evidence and your knowledge. If neither is reliable, answer exactly: I do not know.')

class Runtime:
    def __init__(self, model_path=None, max_input=512, max_new=16):
        configure_cache()
        self.model_path = str(model_path or os.environ.get('SLEA_MODEL_PATH') or ROOT/'models/Qwen2.5-3B-Instruct')
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, use_fast=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path, local_files_only=True, torch_dtype=torch.float16,
            device_map={'': 'cuda:0'}, attn_implementation='eager', low_cpu_mem_usage=True)
        self.model.eval().requires_grad_(False)
        assert all(p.dtype == torch.float16 for p in self.model.parameters())
        self.max_input, self.max_new = max_input, max_new
        self.fixed_length_benchmark=False
        self.calls = 0
        self._count_handle = self.model.register_forward_pre_hook(self._count)

    def _count(self, module, args):
        self.calls += 1

    def encode(self, question, context=None):
        if context is None:
            user = f'Question: {question}'
        else:
            if '<context>' in context or '</context>' in context:
                raise ValueError('Reserved delimiter in context')
            user = f'<context>\n{context}\n</context>\nQuestion: {question}'
        text = self.tokenizer.apply_chat_template(
            [{'role':'system','content':SYSTEM}, {'role':'user','content':user}],
            tokenize=False, add_generation_prompt=True)
        encoded = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded.pop('offset_mapping')
        ids = encoded['input_ids']
        if len(ids) > self.max_input:
            # No silent truncation: answer-bearing evidence and final prompt position retained.
            raise ValueError(f'Input has {len(ids)} tokens, exceeds fixed {self.max_input}')
        span = None
        if context is not None:
            begin = text.index('<context>\n')+len('<context>\n')
            end = begin+len(context)
            indices = [i for i,(a,b) in enumerate(offsets) if b > begin and a < end and b > a]
            if not indices:
                raise ValueError('Empty context token span')
            span = [min(indices), max(indices)+1]
            assert offsets[span[0]][0] <= begin and offsets[span[1]-1][1] >= end
            assert span[1] < len(ids)
        tensors = {'input_ids': torch.tensor([ids],device='cuda'),
                   'attention_mask':torch.ones((1,len(ids)),dtype=torch.long,device='cuda')}
        return tensors, {'context_span': span, 'prompt_tokens':len(ids),
                         'last_prompt_position':len(ids)-1, 'truncated':False, 'padding':'none_batch1'}

    @torch.inference_mode()
    def generate(self, question, context=None, controller=None, cad_alpha=None, capture=None):
        cuda_sync()
        request_start=time.perf_counter()
        inputs, meta = self.encode(question, context)
        if controller:
            controller.reset(meta['context_span'])
        if capture:
            capture.reset()
        base = None
        if cad_alpha is not None:
            base, _ = self.encode(question)
        count_start = self.calls
        generated, confidences = [], []
        cache = base_cache = None
        first = None
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos,int) else eos
        cuda_sync()
        start = time.perf_counter()
        for step in range(self.max_new):
            out = self.model(**inputs, past_key_values=cache, use_cache=True, num_logits_to_keep=1)
            cache = out.past_key_values
            logits = out.logits[:, -1, :]
            if capture and step == 0:
                capture.finish_prefill()
            if cad_alpha is not None:
                ref = self.model(**base, past_key_values=base_cache, use_cache=True, num_logits_to_keep=1)
                base_cache = ref.past_key_values
                logits = (1+cad_alpha)*logits-cad_alpha*ref.logits[:, -1, :]
            probs = logits.float().softmax(-1)  # metric computation, not model/arbitrator precision
            top = probs.topk(2,dim=-1).values[0]
            token = logits.argmax(-1,keepdim=True)
            if step == 0:
                first_stats = {'confidence':float(top[0]),'margin':float(top[0]-top[1]),
                               'entropy':float(-(probs*probs.clamp_min(1e-12).log()).sum())}
                cuda_sync()
                first = (time.perf_counter()-request_start)*1000
            generated.append(int(token.item()))
            confidences.append(float(top[0]))
            if generated[-1] in eos and not self.fixed_length_benchmark:
                break
            inputs = {'input_ids':token,'attention_mask':torch.ones((1,meta['prompt_tokens']+step+1),device='cuda',dtype=torch.long)}
            if base is not None:
                base = {'input_ids':token,'attention_mask':torch.ones((1,base['attention_mask'].shape[1]+1),device='cuda',dtype=torch.long)}
        cuda_sync()
        model_elapsed = (time.perf_counter()-start)*1000
        answer=self.tokenizer.decode(generated,skip_special_tokens=True).strip()
        confidence=float(torch.tensor(confidences).clamp_min(1e-8).log().mean().exp())
        elapsed=(time.perf_counter()-request_start)*1000
        return {'answer':answer,
                'first_token':first_stats,'answer_confidence':confidence,
                'generated_tokens':len(generated),'forward_calls':self.calls-count_start,
                'ttft_ms':first,'latency_ms':elapsed,'peak_allocated_bytes':torch.cuda.max_memory_allocated(),
                'model_generation_ms':model_elapsed,
                'timing_scope':'request tokenization, device input preparation, model generation, decoding; warmed model and registered controller setup excluded',
                'stopping_policy':'fixed token budget, timing only' if self.fixed_length_benchmark else 'EOS or token cap',
                **meta}

class Capture:
    """Capture pre-final-normalization decoder outputs; layers numbered 1..L."""
    def __init__(self, model):
        self.values = {}
        self.active = True
        self.handles = [layer.register_forward_hook(self._hook(i+1)) for i,layer in enumerate(model.model.layers)]

    def _hook(self, layer):
        def hook(module,args,output):
            if self.active:
                self.values[layer] = output[0][:,-1,:].detach().clone()
        return hook

    def reset(self):
        self.values = {}
        self.active = True

    def finish_prefill(self):
        self.active = False

    def array(self):
        return torch.cat([self.values[k] for k in sorted(self.values)],dim=0).cpu().numpy()

    def close(self):
        for handle in self.handles:
            handle.remove()
