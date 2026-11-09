"""Causal prefix-only feature hooks; no future decoder state or paired prefix."""
import time
import torch
import torch.nn.functional as F

GROUPS = {'last': ['last'], 'context': ['context'], 'cross': ['cross2', 'cross4'],
          'focal': ['focal', 'geometry'], 'lens': ['lens', 'geometry'],
          'all_causal': ['context', 'cross2', 'cross4', 'focal', 'geometry', 'lens']}


def ln(x):
    return F.layer_norm(x, (x.shape[-1],), eps=1e-5)


class CausalFeatures:
    def __init__(self, model, layers, group=None):
        self.model, self.layers, self.group = model, sorted(layers), group
        self.handles = [model.register_forward_pre_hook(self.root_input, with_kwargs=True)]
        for layer in sorted(set([2, 4]+self.layers)):
            module = model.model.layers[layer-1]
            if layer in self.layers:
                self.handles.append(module.self_attn.register_forward_pre_hook(self.attention_pre, with_kwargs=True))
                self.handles.append(module.self_attn.register_forward_hook(self.attention_post(layer)))
            self.handles.append(module.register_forward_hook(self.decoder_post(layer)))
        self.reset(None)

    def reset(self, span):
        self.span = span; self.active = span is not None
        self.early = {}; self.alpha = {}; self.values = {}; self.raw = {}
        self.lens_events = []; self.lens_calls = 0; self.trace = []; self.ids = None

    def root_input(self, module, args, kwargs):
        if self.active and self.ids is None:
            self.ids = kwargs['input_ids'][0].detach()

    def attention_pre(self, module, args, kwargs):
        if self.active:
            kwargs = dict(kwargs); kwargs['output_attentions'] = True
        return args, kwargs

    def attention_post(self, layer):
        def hook(module, args, output):
            if self.active:
                assert output[1] is not None and output[1].dtype == torch.float16
                a, b = self.span
                self.alpha[layer] = output[1][0, :, -1, a:b].mean(0).detach()
                self.trace.append(['attention', layer])
        return hook

    def early_lens(self, layer, vectors, alpha):
        device = self.model.lm_head.weight.device
        head_input = vectors.to(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        with torch.cuda.device(device):
            start.record()
            logits = self.model.lm_head(self.model.model.norm(head_input))
            assert logits.dtype == torch.float16
            prob = logits.softmax(-1)
            top_logits = logits.topk(2, -1).values
            top_p = prob.topk(2, -1).values
            ids = logits.argmax(-1)
            a, b = self.span
            context_ids = self.ids[a:b].to(device)
            matches = ids[:, None] == context_ids[None]
            support = (matches.to(torch.float16)*alpha.to(device)[None]).sum(-1)
            entropy = -(prob*prob.clamp_min(torch.finfo(torch.float16).tiny).log()).sum(-1)
            stats = torch.stack([top_logits[:, 0]-top_logits[:, 1], top_p[:, 0]-top_p[:, 1], entropy,
                top_p[:, 0], matches.any(-1).to(torch.float16), support], -1).reshape(-1)
            end.record()
        self.lens_events.append((layer, start, end)); self.lens_calls += 1
        self.trace.append(['early_lens', layer])
        return stats.to(vectors.device)

    def decoder_post(self, layer):
        def hook(module, args, output):
            if not self.active:
                return
            h = output[0][0]
            if layer in [2, 4]:
                self.early[layer] = h[-1].detach().clone()
            if layer not in self.layers:
                return
            a, b = self.span; last = h[-1]; mean = h[a:b].mean(0)
            alpha = self.alpha[layer].to(h.device)
            mass = alpha.sum(); focal_raw = alpha@h[a:b]
            focal = focal_raw/mass.clamp_min(torch.finfo(torch.float16).tiny)
            old2, old4 = self.early[2].to(h.device), self.early[4].to(h.device)
            cosine = lambda x,y: F.cosine_similarity(x[None], y[None], dim=-1)[0]
            normalized_alpha = alpha/mass.clamp_min(torch.finfo(torch.float16).tiny)
            geometry = torch.stack([cosine(last,mean), cosine(last,focal), cosine(last,old2), cosine(last,old4), mass,
                -(normalized_alpha*normalized_alpha.clamp_min(torch.finfo(torch.float16).tiny).log()).sum()])
            values = dict(last=ln(last), context=ln(last)-ln(mean), cross2=ln(last-old2), cross4=ln(last-old4),
                          focal=ln(last)-ln(focal), geometry=geometry)
            if self.group is None or 'lens' in GROUPS[self.group]:
                values['lens'] = self.early_lens(layer, torch.stack([last,focal]), alpha)
            self.values[layer] = values
            self.raw[layer] = dict(last=last.detach().clone(), context_mean=mean.detach().clone(), focal_raw=focal_raw.detach().clone())
            assert all(v.dtype == torch.float16 and bool(torch.isfinite(v).all()) for v in values.values())
            self.trace.append(['features_complete', layer])
            if layer == self.layers[-1]:
                self.active = False
        return hook

    def vector(self, layer, group):
        return torch.cat([self.values[layer][key] for key in GROUPS[group]])

    def audit(self):
        return dict(causal_max_layers=self.layers, trace=self.trace, auxiliary_head_projections=self.lens_calls,
            projected_positions=2*self.lens_calls, lens_ms=[dict(layer=l, ms=float(a.elapsed_time(b))) for l,a,b in self.lens_events],
            backbone_replays=0, future_layer_features=False)

    def close(self):
        for handle in self.handles:
            handle.remove()
