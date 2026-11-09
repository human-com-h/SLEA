"""Mathematically equivalent fp16 eager QK scaling order, for overflow diagnostics."""
import hashlib
import inspect
import textwrap
import types
import torch


def disable_reduced_accumulation():
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    if hasattr(torch.backends.cuda.matmul, 'allow_fp16_accumulation'):
        torch.backends.cuda.matmul.allow_fp16_accumulation = False


def prescale_query(model):
    import transformers.models.qwen2.modeling_qwen2 as module
    original = textwrap.dedent(inspect.getsource(module.Qwen2Attention.forward))
    old = 'attn_weights = torch.matmul(query_states, key_states.transpose(2, 3)) / math.sqrt(self.head_dim)'
    new = 'attn_weights = torch.matmul(query_states / math.sqrt(self.head_dim), key_states.transpose(2, 3))'
    assert original.count(old) == 1
    patched = original.replace(old, new); namespace = {}
    exec(compile(patched, '<phase4-fp16-eager-query-prescale>', 'exec'), module.__dict__, namespace)
    forward = namespace['forward']
    for decoder in model.model.layers:
        attention = decoder.self_attn
        if hasattr(attention, '_old_forward'): attention._old_forward = types.MethodType(forward, attention)
        else: attention.forward = types.MethodType(forward, attention)
    return {'change': 'Q/sqrt(head_dim) before fp16 QK matmul instead of scaling already-rounded fp16 QK output',
            'original_source_sha256': hashlib.sha256(original.encode()).hexdigest(),
            'patched_source_sha256': hashlib.sha256(patched.encode()).hexdigest(),
            'model_weights_and_activation_output_dtype': 'float16', 'attention_backend': 'eager',
            'extra_model_calls': 0, 'preserves_existing_accelerate_wrappers_and_mask_hooks': True}


def finite_observers(model):
    observed = []; handles = []
    def hook(name):
        def callback(module, args, output):
            t = output[0] if isinstance(output, tuple) else output
            if not isinstance(t, torch.Tensor): return
            if len(observed) >= 200: return
            finite = torch.isfinite(t)
            observed.append({'name': name, 'dtype': str(t.dtype), 'device': str(t.device),
                             'nonfinite': int((~finite).sum()),
                             'max_abs_finite': float(t[finite].abs().max()) if finite.any() else None})
        return callback
    for i, layer in enumerate(model.model.layers):
        for name, module in [('q_proj', layer.self_attn.q_proj), ('k_proj', layer.self_attn.k_proj),
                             ('attention', layer.self_attn), ('mlp', layer.mlp), ('layer', layer)]:
            handles.append(module.register_forward_hook(hook(f'{i+1}/{name}')))
    return observed, handles
