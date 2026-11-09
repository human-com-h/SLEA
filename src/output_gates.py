"""Cheap refusal gates and the uploaded Sad26 interaction formula.

Interaction Score is replicated as alpha(q)*(1-max_sentence clamp(cos,0,1)).
Its downstream binary source selection is our validation-fitted evaluation policy.
"""
from __future__ import annotations
import os
from pathlib import Path
import re
import time
import numpy as np
import torch
from .common import ROOT,cuda_sync
from .metrics import summarize

class InteractionScore:
    def __init__(self,path=None):
        from sentence_transformers import SentenceTransformer
        path=path or os.environ.get('SLEA_ENCODER_PATH') or ROOT/'models/all-MiniLM-L6-v2'
        if not Path(path).exists():
            raise FileNotFoundError('Mount the MiniLM encoder or fetch it inside Kaggle; no implicit local download')
        self.encoder=SentenceTransformer(str(path),device='cuda',local_files_only=True)
        self.encoder.half().eval()
        self.encoder.encode(['warm up'],convert_to_tensor=True,normalize_embeddings=True,show_progress_bar=False)

    @torch.inference_mode()
    def score(self,answer,context,alpha):
        sentences=[s.strip() for s in re.split(r'(?<=[.!?])\s+|\n+',context) if s.strip()]
        cuda_sync();start=time.perf_counter()
        vectors=self.encoder.encode([answer]+sentences,convert_to_tensor=True,
            normalize_embeddings=True,show_progress_bar=False,batch_size=32)
        phi=(vectors[1:]@vectors[0]).max().clamp(0,1)
        value=float((alpha*(1-phi)).item())
        cuda_sync()
        return value,(time.perf_counter()-start)*1000

def select_thresholds(vanilla,closed_by_fact,encoder,source_rows):
    if any(r['split']!='val' for r in vanilla):raise ValueError('Threshold selection requires validation rows')
    lookup={r['id']:r for r in source_rows}
    thresholds={}
    for name,field,larger in [('entropy','entropy',True),('margin','margin',False)]:
        signal=np.array([r['first_token'][field] for r in vanilla])
        trials=[]
        for threshold in [float('-inf'),*np.quantile(signal,[.1,.25,.5,.75,.9]).tolist(),float('inf')]:
            predicted=[]
            for r,s in zip(vanilla,signal):
                item=dict(r)
                if (s>threshold) if larger else (s<threshold):item['answer']='I do not know'
                predicted.append(item)
            summary=summarize(predicted)
            utility=(summary['h_score'] or 0,summary['refusal_f1'],summary['align_accuracy'] or 0)
            trials.append((utility,threshold))
        best=max(trials,key=lambda x:x[0])[1]
        # JSON-safe sentinels are translated back during evaluation.
        thresholds[name]=best if np.isfinite(best) else ('positive_infinity' if best>0 else 'negative_infinity')
    scored=[]
    for r in vanilla:
        raw=lookup[r['id']]
        cb=closed_by_fact[raw['fact_id']]
        score,ms=encoder.score(cb['answer'],raw['context'],cb['first_token']['confidence'])
        scored.append((r,cb,score,ms))
    signal=np.array([x[2] for x in scored])
    trials=[]
    for threshold in [-1.,*np.quantile(signal,[.1,.25,.5,.75,.9]).tolist(),2.]:
        predicted=[]
        for rag,cb,s,ms in scored:
            item=dict(rag)
            if s>threshold:item['answer']=cb['answer']
            predicted.append(item)
        summary=summarize(predicted)
        trials.append({'threshold':threshold,'h_score':summary['h_score'] or 0})
    thresholds['interaction']=max(trials,key=lambda r:r['h_score'])['threshold']
    return thresholds,{'interaction_trials':trials,'interaction_val_scores':[
        {'id':r['id'],'score':s,'encoding_ms':ms} for r,cb,s,ms in scored]}

def threshold_value(value):
    return float('inf') if value=='positive_infinity' else float('-inf') if value=='negative_infinity' else float(value)
