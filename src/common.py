from __future__ import annotations
import hashlib
import json
import os
import random
import re
import string
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
STATES = ['ext', 'param', 'align', 'abstain']
SEEDS = [42, 3407, 1729]

def seed_all(seed: int):
    if seed not in SEEDS:
        raise ValueError(f'Seed must be one of {SEEDS}')
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    return {'python': seed, 'numpy': seed, 'torch': seed, 'cuda': seed,
            'cudnn_benchmark': False, 'allow_tf32': False,
            'deterministic_algorithms': False}

def dump(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8').splitlines() if line.strip()]

def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False)+'\n' for r in rows), encoding='utf-8')

def fingerprint(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]

def normalize(text):
    text = str(text).lower().strip()
    text = re.sub(r'^(answer|the answer is)\s*:\s*', '', text)
    text = ''.join(ch for ch in text if ch not in string.punctuation)
    return ' '.join(re.sub(r'\b(a|an|the)\b', ' ', text).split())

def exact_match(answer, aliases):
    value = normalize(answer)
    return bool(value) and value in {normalize(a) for a in aliases if normalize(a)}

def contains_alias(text, aliases):
    value = ' '+normalize(text)+' '
    return any(' '+normalize(a)+' ' in value for a in aliases if len(normalize(a)) >= 3)

def is_refusal(answer):
    return bool(re.search(r"\b(i (do not|don't) know|unknown|cannot (answer|determine)|insufficient (information|evidence))\b", answer.lower()))

def cuda_sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()

def configure_cache():
    # Current process only, never user/system environment.
    os.environ.setdefault('HF_HOME', str(ROOT/'.cache/huggingface'))
    os.environ.setdefault('HF_HUB_DISABLE_XET', '1')
