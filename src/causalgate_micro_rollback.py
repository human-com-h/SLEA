"""Replay only speculative decoder layers inside the original prefill."""
import torch

from .phase4_runtime import ScopeController


class MicroRollback(ScopeController):
    def __init__(self, runtime, control, read, collector, decide, *, first_masked=None):
        self.read = read
        self.collector = collector
        self.decide_online = decide
        self.first_masked = control + 1 if first_masked is None else first_masked
        self.layers = runtime.model.model.layers
        assert 1 <= self.first_masked <= read + 1 <= len(self.layers)
        super().__init__(runtime, self.first_masked - 1)
        self.extra_handles = []
        for index, layer in enumerate(self.layers):
            self.extra_handles.append(layer.register_forward_pre_hook(
                self.remember(index), with_kwargs=True))
        self.extra_handles.append(self.layers[read - 1].register_forward_hook(self.finish))
        self.extra_handles.append(runtime.model.register_forward_hook(self.record_cache))
        self.reset(None)

    def _middle(self, module, args, output):
        return output

    def reset(self, span):
        super().reset(span)
        self.collector.reset(span)
        self.pending = span is not None
        self.arguments = {}
        self.calls_by_layer = [0] * len(self.layers)
        self.replay_events = []
        self.cache_steps = []
        self.discarded_layers = 0

    def remember(self, index):
        def hook(module, args, kwargs):
            if self.span is None:
                return
            self.calls_by_layer[index] += 1
            if self.pending and self.first_masked - 1 <= index < self.read:
                self.arguments[index] = (args, dict(kwargs))
        return hook

    def record_cache(self, module, args, output):
        if self.span is not None:
            cache = output.past_key_values
            keys = cache.key_cache if hasattr(cache, 'key_cache') else [item[0] for item in cache]
            self.cache_steps.append([int(key.shape[-2]) for key in keys])

    def finish(self, module, args, output):
        if not self.pending:
            return output
        self.pending = False
        self.collector.active = False
        self.block = bool(self.decide_online(self.collector))
        self.gate_calls += 1
        arguments, self.arguments = self.arguments, {}
        if not self.block or not arguments:
            return output
        start = self.first_masked - 1
        args, kwargs = arguments[start]
        hidden = args[0] if args else kwargs['hidden_states']
        cache = kwargs['past_key_value']
        assert len(cache.key_cache) == len(cache.value_cache) == self.read
        assert all(values[1]['past_key_value'] is cache for values in arguments.values())
        del cache.key_cache[start:]
        del cache.value_cache[start:]
        self.discarded_layers = self.read - start
        for index in range(start, self.read):
            layer = self.layers[index]
            args, kwargs = arguments[index]
            if args:
                args = (hidden, *args[1:])
            else:
                kwargs['hidden_states'] = hidden
            device = next(layer.parameters()).device
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    begin = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    begin.record()
            output = layer(*args, **kwargs)
            hidden = output[0]
            if device.type == 'cuda':
                with torch.cuda.device(device):
                    end.record()
                self.replay_events.append((begin, end))
        return output

    def close(self):
        for handle in self.extra_handles:
            handle.remove()
        super().close()
