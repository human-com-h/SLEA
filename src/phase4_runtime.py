"""New-model-only fp16 two-GPU runtime. Does not modify sealed 3B runtime."""
import hashlib
import json
import os
import math
import time
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache
from .common import ROOT, dump
from .model_runtime import Runtime
from .early_controller import FoldedGate
from .slea_controller import SLEAController


def synchronize():
    for i in range(torch.cuda.device_count()): torch.cuda.synchronize(i)


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for part in iter(lambda: f.read(8*1024*1024), b''): h.update(part)
    return h.hexdigest()


def bootstrap(cfg):
    if not Path('/kaggle/working').is_dir(): raise RuntimeError('Cloud-only weight access')
    from huggingface_hub import snapshot_download
    source = json.loads((ROOT/'configs/phase4_qwen7b_source.json').read_text())
    matches = []
    for p in Path('/kaggle/input').rglob('config.json'):
        try: c = json.loads(p.read_text())
        except (ValueError, OSError): continue
        if c.get('model_type') == source['config']['model_type'] and c.get('hidden_size') == source['config']['hidden_size'] and c.get('num_hidden_layers') == source['config']['num_hidden_layers']:
            if not c.get('quantization_config') and all((p.parent/n).is_file() for n in source['weight_sha256']): matches.append(p.parent)
    assert len(matches) == 1, 'Exactly one official unquantized matching model must be mounted'
    mounted = matches[0]
    for name, digest in source['weight_sha256'].items(): assert file_sha(mounted/name) == digest, name
    target = Path('/kaggle/temp/phase4_model'); target.mkdir(parents=True, exist_ok=True)
    snapshot_download(cfg['model'], revision=cfg['revision'], local_dir=target,
        allow_patterns=['config.json', 'generation_config.json', 'tokenizer*.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt', 'model.safetensors.index.json'])
    for name, digest in source['metadata_sha256'].items(): assert file_sha(target/name) == digest, name
    for name in source['weight_sha256']:
        link = target/name
        if not link.exists(): link.symlink_to(mounted/name)
    os.environ['SLEA_MODEL_PATH'] = str(target)
    dump(ROOT/'results/phase4_model_manifest.json', {'model': cfg['model'], 'revision': cfg['revision'],
         'mounted_path': str(mounted), 'runtime_path': str(target), 'weights_sha256': source['weight_sha256'],
         'metadata_sha256': source['metadata_sha256'], 'precision': 'original weights explicitly loaded as float16',
         'credential_files_uploaded': False})


class Phase4Runtime(Runtime):
    def __init__(self, cfg):
        assert torch.cuda.device_count() == cfg['gpu_count'] == 2, 'Two GPUs required; no silent single-GPU fallback'
        self.model_path = os.environ['SLEA_MODEL_PATH']
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, local_files_only=True, use_fast=True)
        config = json.loads((Path(self.model_path)/'config.json').read_text())
        n = config['num_hidden_layers']
        mapping = {'model.embed_tokens': 0, 'model.rotary_emb': 0, 'model.norm': 1, 'lm_head': 1}
        mapping.update({f'model.layers.{i}': 0 if i < n//2 else 1 for i in range(n)})
        self.model = AutoModelForCausalLM.from_pretrained(self.model_path, local_files_only=True,
            torch_dtype=torch.float16, device_map=mapping, attn_implementation='eager', low_cpu_mem_usage=True)
        self.model.eval().requires_grad_(False)
        assert all(p.dtype == torch.float16 and p.device.type == 'cuda' for p in self.model.parameters())
        assert {p.device.index for p in self.model.parameters()} == {0, 1}
        numerical_path = ROOT/'configs/phase4_qwen7b_numerics.json'
        self.require_finite = numerical_path.exists()
        numerical = json.loads(numerical_path.read_text()) if numerical_path.exists() else {'technical_diagnostic_unfixed': True}
        if numerical.get('disable_reduced_accumulation'):
            from .phase4_numerics import disable_reduced_accumulation
            disable_reduced_accumulation()
        if numerical.get('query_prescale'):
            from .phase4_numerics import prescale_query
            actual = prescale_query(self.model)
            assert actual == numerical['implementation']
        self.max_input, self.max_new = cfg['max_input_tokens'], cfg['max_new_tokens']
        self.fixed_length_benchmark = False; self.calls = 0
        self._count_handle = self.model.register_forward_pre_hook(self._count)
        self.model._phase4_controllers = []
        self.capture_first_logits = False; self.last_first_logits = None
        dump(ROOT/'results/phase4_runtime_environment.json', {'device_map': self.model.hf_device_map,
             'devices': [{'name': torch.cuda.get_device_name(i), 'total_bytes': torch.cuda.get_device_properties(i).total_memory} for i in range(2)],
             'torch': torch.__version__, 'all_model_parameters_frozen_fp16': True,
             'numerical_policy': numerical,
             'synchronization': 'all visible GPUs before/after measured operations'})

    @torch.inference_mode()
    def generate(self, question, context=None, controller=None, capture=None, cad_alpha=None):
        for c in self.model._phase4_controllers: c.reset(None)
        if capture is not None: capture.reset()
        synchronize(); request_start = time.perf_counter()
        count_start = self.calls
        inputs, meta = self.encode(question, context)
        if controller is not None: controller.reset(meta['context_span'])
        base = None
        if cad_alpha is not None: base, _ = self.encode(question)
        cache = DynamicCache(); base_cache = DynamicCache() if base is not None else None
        generated = []; confidences = []; first = None
        eos = self.model.generation_config.eos_token_id
        eos = [eos] if isinstance(eos, int) else eos
        stop = 'fixed_budget' if self.fixed_length_benchmark else 'token_cap'
        synchronize(); model_start = time.perf_counter()
        for step in range(self.max_new):
            out = self.model(**inputs, past_key_values=cache, use_cache=True, num_logits_to_keep=1)
            cache = out.past_key_values; logits = out.logits[:, -1, :]
            if capture is not None and step == 0: capture.finish_prefill()
            if base is not None:
                ref = self.model(**base, past_key_values=base_cache, use_cache=True, num_logits_to_keep=1)
                base_cache = ref.past_key_values
                logits = (1+cad_alpha)*logits-cad_alpha*ref.logits[:, -1, :]
            if step == 0 and self.capture_first_logits: self.last_first_logits = logits.detach().float().cpu()
            probs = logits.float().softmax(-1); top = probs.topk(2, dim=-1).values[0]
            token = logits.argmax(-1, keepdim=True)
            if step == 0:
                first_stats = {'confidence': float(top[0]), 'margin': float(top[0]-top[1]),
                               'entropy': float(-(probs*probs.clamp_min(1e-12).log()).sum())}
                synchronize(); first = (time.perf_counter()-request_start)*1000
            generated.append(int(token.item())); confidence_value = float(top[0])
            if self.require_finite and not math.isfinite(confidence_value):
                raise FloatingPointError('Nonfinite logits/probabilities in fixed fp16 runtime; no sample exclusion or silent repair')
            confidences.append(confidence_value)
            if generated[-1] in eos and not self.fixed_length_benchmark:
                stop = 'eos'; break
            token = token.to('cuda:0')
            inputs = {'input_ids': token, 'attention_mask': torch.ones((1, meta['prompt_tokens']+step+1), device='cuda:0', dtype=torch.long)}
            if base is not None: base = {'input_ids': token, 'attention_mask': torch.ones((1, base['attention_mask'].shape[1]+1), device='cuda:0', dtype=torch.long)}
        synchronize(); model_ms = (time.perf_counter()-model_start)*1000
        lengths = [int(k.shape[-2]) for k in cache.key_cache]
        assert len(lengths) == len(self.model.model.layers) and set(lengths) == {meta['prompt_tokens']+len(generated)-1}
        answer = self.tokenizer.decode(generated, skip_special_tokens=True).strip()
        confidence = float(np.exp(np.log(np.maximum(confidences, 1e-8)).mean()))
        return {'answer': answer, 'first_token': first_stats, 'answer_confidence': confidence,
            'generated_tokens': len(generated), 'generated_token_ids': generated, 'stop_reason': stop, 'eos_token_ids': eos,
            'forward_calls': self.calls-count_start, 'ttft_ms': first, 'latency_ms': (time.perf_counter()-request_start)*1000,
            'model_generation_ms': model_ms, 'stopping_policy': 'fixed token budget, timing only' if self.fixed_length_benchmark else 'EOS or token cap',
            'peak_allocated_bytes': sum(torch.cuda.max_memory_allocated(i) for i in range(2)),
            'peak_allocated_bytes_by_gpu': [torch.cuda.max_memory_allocated(i) for i in range(2)],
            'cache_lengths': lengths, 'cache_devices': [str(k.device) for k in cache.key_cache],
            'timing_scope': 'two-GPU synchronized full request including tokenization, transfers, greedy decode and text decoding', **meta}


class Phase4Gate(FoldedGate):
    def decide(self, values):
        with torch.cuda.device(self.weight.device): return super().decide(values)


class ScopeController(SLEAController):
    def __init__(self, runtime, layer, scope='all_queries', gate=None, force_route=None, soft_bias=0.):
        self.gate = gate; self.scope = scope; self.soft_bias = soft_bias
        self.logits = None; self.block = False; self.host_ms = 0.
        super().__init__(runtime.model, None, layer, channel1=False, channel2=True,
                         residual_strength=0., span_bias=soft_bias, force_route=force_route)
        runtime.model._phase4_controllers.append(self)

    def reset(self, span):
        super().reset(span); self.logits = None; self.block = False; self.host_ms = 0.

    def _middle(self, module, args, output):
        if self.span is None: return output
        h = output[0]; self.middle_calls += 1; self.query_lengths.append(h.shape[1])
        if self.route is None:
            t = time.perf_counter()
            if self.gate is not None: self.route, self.probabilities, self.logits = self.gate.decide([h[:, -1, :]])
            else:
                self.route = torch.tensor([self.force_route], dtype=torch.long, device=h.device)
                self.probabilities = torch.nn.functional.one_hot(self.route, 4).to(torch.float16)
            self.block = bool((self.route == 1).item()); self.host_ms = (time.perf_counter()-t)*1000; self.gate_calls += 1
        return output

    def _attention(self, module, args, kwargs):
        if self.span is None or not self.block: return args, kwargs
        mask = kwargs.get('attention_mask')
        assert mask is not None and mask.ndim == 4 and mask.dtype == torch.float16
        start, end = self.span; assert 0 <= start < end <= mask.shape[-1]
        changed = mask.clone()
        selection = (slice(None), slice(None), slice(None) if self.scope == 'all_queries' else slice(-1, None), slice(start, end))
        if self.soft_bias: changed[selection] -= self.soft_bias
        else: changed[selection] = -float('inf')
        kwargs = dict(kwargs); kwargs['attention_mask'] = changed; self.attention_calls += 1
        return args, kwargs

    def audit(self):
        result = super().audit()
        result.update(block=self.block, attention_scope=self.scope, eta=0., extra_model_calls=0,
                      gate_host_ms=self.host_ms, folded_four_logits=self.logits[0].float().cpu().tolist() if self.logits is not None else None,
                      control_type='soft_span' if self.soft_bias else 'hard', gate_device=str(self.route.device) if self.route is not None else None)
        return result

    def close(self):
        super().close()
        self.model._phase4_controllers.remove(self)


class FeatureCapture:
    def __init__(self, runtime, layers):
        self.layers = layers; self.values = {}; self.active = True
        self.handles = [runtime.model.model.layers[l-1].register_forward_hook(self._hook(l)) for l in layers]

    def _hook(self, layer):
        def hook(module, args, output):
            if self.active: self.values[layer] = output[0][0, -1].detach().cpu().numpy().copy()
        return hook

    def reset(self): self.values = {}; self.active = True
    def finish_prefill(self): self.active = False
    def array(self): return np.stack([self.values[l] for l in self.layers])
    def close(self):
        for h in self.handles: h.remove()
