"""Budget-matched CAST/ContextFocus task adaptations, no attention modification.

Coordinates are residual AFTER decoder l (1-based). This is the input of
decoder l+1 in the author's pre-decoder CAST implementation. Offline fits
use float64; online activations and vectors preserve the model's float16.
"""
from dataclasses import dataclass
import numpy as np
import torch
from sklearn.decomposition import PCA


def pairwise_pca(positive, negative):
    """Author's pca_pairwise: pair-center, PC1, orient toward positive."""
    positive, negative = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    if positive.shape != negative.shape or positive.ndim != 2 or len(positive) < 2:
        raise ValueError('At least two shape-matched contrast pairs required')
    delta = (positive - negative) / 2
    train = np.stack([delta, -delta], axis=1).reshape(-1, delta.shape[1])
    v = PCA(n_components=1, svd_solver='full', random_state=42).fit(train).components_[0]
    if np.mean((positive - negative) @ v) < 0:
        v = -v
    return v.astype(np.float32)


def contextfocus_direction(positive, negative):
    p, n = np.asarray(positive, dtype=np.float64), np.asarray(negative, dtype=np.float64)
    if p.shape != n.shape or not len(p):
        raise ValueError('Paired source calibration activations required')
    return (p - n).mean(axis=0).astype(np.float32)


def condition_score(hidden, direction):
    """Rank-one Pc h, tanh, cosine; O(d), no dense d-by-d projector."""
    c = direction.to(hidden.device, hidden.dtype)
    projected = c * ((hidden * c).sum(-1, keepdim=True) / c.square().sum().clamp_min(1e-7))
    return torch.nn.functional.cosine_similarity(hidden, projected.tanh(), dim=-1, eps=1e-7)


def fit_condition_threshold(scores, labels):
    """Calibration-only balanced accuracy, deterministic ties, no generation grid."""
    s, y = np.asarray(scores, dtype=float), np.asarray(labels, dtype=bool)
    if not np.isfinite(s).all() or y.all() or not y.any():
        raise ValueError('Both condition classes and finite scores required')
    unique = np.unique(s)
    thresholds = np.r_[np.nextafter(unique[0], -np.inf), (unique[1:] + unique[:-1]) / 2,
                       np.nextafter(unique[-1], np.inf)]
    best = None
    for comparator in ['greater', 'less']:
        for threshold in thresholds:
            pred = s > threshold if comparator == 'greater' else s < threshold
            balanced = (pred[y].mean() + (~pred[~y]).mean()) / 2
            candidate = {'threshold': float(threshold), 'comparator': comparator,
                         'calibration_balanced_accuracy': float(balanced)}
            if best is None or balanced > best['calibration_balanced_accuracy']:
                best = candidate
    return best


@dataclass(frozen=True)
class SteeringSpec:
    method: str
    layer: int
    alpha: float
    threshold: float = 0.0
    comparator: str = 'greater'
    ooi_normalize: bool = True


class ResidualController:
    """Request-local condition, reused through ordinary KV-cached decoding."""
    def __init__(self, model, spec, behavior, condition=None):
        if spec.method not in ['cast_rag', 'cast_refusal', 'contextfocus']:
            raise ValueError(spec.method)
        if not 1 <= spec.layer < len(model.model.layers):
            raise ValueError('Control must precede another decoder layer')
        self.spec = spec
        self.behavior = torch.as_tensor(np.array(behavior, copy=True))
        self.condition = None if condition is None else torch.as_tensor(np.array(condition, copy=True))
        if spec.method.startswith('cast') and condition is None:
            raise ValueError('CAST requires its own PCA condition, never a SLEA gate')
        layer = model.model.layers[spec.layer-1]
        device = next(layer.parameters()).device
        self.behavior = self.behavior.to(device=device, dtype=torch.float16)
        if self.condition is not None:
            self.condition = self.condition.to(device=device, dtype=torch.float16)
        self.handle = layer.register_forward_hook(self._hook)
        self.reset()

    def reset(self, span=None):
        self.calls = self.condition_calls = self.edits = 0
        self.active = self.spec.method == 'contextfocus'
        self.score = None
        self.query_lengths = []

    def _hook(self, module, args, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.calls += 1
        self.query_lengths.append(int(hidden.shape[1]))
        if self.calls == 1 and self.condition is not None:
            self.condition_calls += 1
            self.score = float(condition_score(hidden[0, -1], self.condition))
            self.active = (self.score > self.spec.threshold if self.spec.comparator == 'greater'
                           else self.score < self.spec.threshold)
        if not self.active or self.spec.alpha == 0:
            return output
        changed = hidden.clone()
        target = hidden if self.spec.method.startswith('cast') else hidden[:, -1:, :]
        value = target + self.spec.alpha * self.behavior.to(target.device, target.dtype)
        if self.spec.method.startswith('cast') and self.spec.ooi_normalize:
            oldnorm = target.norm(dim=-1, keepdim=True)
            newnorm = value.norm(dim=-1, keepdim=True)
            if bool((newnorm > oldnorm).any()):
                value = value * (oldnorm / newnorm.clamp_min(torch.finfo(value.dtype).tiny))
        if not bool(torch.isfinite(value).all()):
            raise FloatingPointError('Nonfinite residual steering; no silent fallback')
        if self.spec.method.startswith('cast'):
            changed[:] = value
        else:
            changed[:, -1:, :] = value
        self.edits += 1
        return (changed,) + output[1:] if isinstance(output, tuple) else changed

    def audit(self):
        return {'method': self.spec.method, 'layer': self.spec.layer, 'alpha': self.spec.alpha,
                'condition_calls': self.condition_calls, 'condition_score': self.score,
                'condition_active': self.active, 'residual_calls': self.calls, 'residual_edits': self.edits,
                'query_lengths': self.query_lengths, 'attention_edits': 0, 'manual_refusal_replacements': 0,
                'token_scope': 'all_prefill_plus_decode' if self.spec.method.startswith('cast') else 'prefill_last_plus_decode',
                'coordinate': 'post_decoder_1_based', 'decision_reused_for_decode': True,
                'extra_model_calls': 0, 'prefix_replays': 0}

    def close(self):
        self.handle.remove()


class FirstTokenTopK:
    """Observe the first existing LM-head output. Never calls the model."""
    def __init__(self, model):
        self.values = None
        self.handle = model.lm_head.register_forward_hook(self._hook)

    def _hook(self, module, args, output):
        if self.values is None:
            logits = output[0, -1].detach()
            values, ids = logits.topk(min(10, logits.numel()))
            self.values = {'token_ids': ids.cpu().tolist(), 'logits': values.float().cpu().tolist(),
                           'logits_dtype': str(logits.dtype), 'kind': 'raw_first_context_forward_lm_head'}

    def reset(self):
        self.values = None

    def close(self):
        self.handle.remove()
