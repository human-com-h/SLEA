"""Calibration-only linear probes folded into one four-score map per available layer.

CPU selection is a counterfactual cache screen, never a substitute for actual
generation. Layers 10/14 use a neighboring-layer heuristic until cloud evaluation.
"""
from __future__ import annotations
import argparse
import hashlib
import itertools
import json
import warnings
from pathlib import Path
import numpy as np
from scipy.special import softmax
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.exceptions import ConvergenceWarning
from threadpoolctl import threadpool_limits
from .common import ROOT, STATES, dump, exact_match, fingerprint, is_refusal, read_jsonl, seed_all


def source_layers(layer, mode):
    return [layer] if mode == 'single' else sorted(set([*range(4, layer + 1, 4), layer]))


def features(x, layers, mode):
    pieces = [np.asarray(x[:, l-1], dtype=np.float64) for l in layers]
    if mode == 'difference':
        pieces = [pieces[0]] + [b-a for a, b in zip(pieces, pieces[1:])]
    return np.concatenate(pieces, axis=1)


def fold_map(weight, bias, center, scale, layers, mode, components=None, whitening=None):
    # Rows are the four classes; absorb PCA, whitening, normalization and centering.
    if components is not None:
        weight = (weight / whitening[None, :]) @ components
    weight = weight / scale[None, :]
    bias = bias - weight @ center
    blocks = weight.reshape(4, len(layers), -1).transpose(1, 2, 0).copy()
    if mode == 'difference':
        old = blocks.copy()
        blocks[:-1] = old[:-1] - old[1:]
    return blocks, bias


def route_probabilities(prob, policy):
    prob=np.asarray(prob,dtype=np.float16)
    adjusted = prob * np.array([1., policy['param_cost'], 1., policy['abstain_cost']],dtype=np.float16)
    adjusted = (adjusted / adjusted.sum(axis=-1, keepdims=True)).astype(np.float16)
    block = (adjusted[:, 1] >= policy['param_threshold']) & (adjusted.argmax(-1) == 1)
    refuse = (~block) & (adjusted[:, 3] >= policy['abstain_threshold']) & (adjusted.argmax(-1) == 3)
    return np.where(block, 1, np.where(refuse, 3, np.where(prob[:, 2] > prob[:, 0], 2, 0)))


def shrink_lda(x, y, shrinkage):
    """Pooled within-class covariance shrunk toward spherical covariance, dual solve.

    Avoid a 10240-square covariance allocation; Woodbury is algebraically exact.
    The positive spherical term makes the system SPD without pseudoinverse.
    """
    means = np.stack([x[y == k].mean(0) for k in range(4)])
    residual = x - means[y]
    n, d = x.shape
    variance = float(np.sum(residual * residual) / (n*d))
    ridge = shrinkage * variance
    u = residual * np.sqrt((1-shrinkage) / n)
    system = np.eye(n) + (u @ u.T) / ridge
    chol = np.linalg.cholesky(system)
    rhs = u @ means.T
    solved = np.linalg.solve(chol.T, np.linalg.solve(chol, rhs))
    weight = (means.T/ridge - u.T @ solved/(ridge*ridge)).T
    bias = -.5 * np.sum(means*weight, axis=1) + np.log(np.bincount(y)/n)
    return weight, bias, {'shrinkage':shrinkage, 'spherical_variance':variance,
        'dual_cholesky_min_diagonal':float(np.diag(chol).min()), 'positive_definite':True}


def load_oracle(folder):
    folder = Path(folder)
    trials = json.loads((folder/'oracle_scan.json').read_text())
    result = {}
    for trial in trials:
        cfg = trial['config']
        if cfg['kind'] == 'none' and cfg['span_mode'] == 'hard':
            result[cfg['layer']] = {r['id']:r for r in read_jsonl(folder/'scan'/(trial['cache_key']+'.jsonl'))}
    return result


def screening_grid(prob, rows, layer, oracle, protocol):
    y = np.array([STATES.index(r['state']) for r in rows])
    base = np.array([exact_match(r['answer'], r['gold_answers']) for r in rows])
    base_refusal = np.array([is_refusal(r['answer']) for r in rows])
    # Unknown interventions are bounded as potentially wrong, never assumed harmless.
    param_ok = np.zeros(len(rows), dtype=bool)
    available = [layer] if layer in oracle else ([8, 12] if layer == 10 else [12, 16])
    for i, row in enumerate(rows):
        if row['state'] == 'param':
            param_ok[i] = all(exact_match(oracle[l][row['id']]['answer'], row['gold_answers']) for l in available)
    thresholds = list(itertools.product(protocol['param_thresholds'], protocol['abstain_thresholds']))
    pt = np.array([t[0] for t in thresholds])[:, None]
    at = np.array([t[1] for t in thresholds])[:, None]
    records = []
    for pc, ac in itertools.product(protocol['param_costs'], protocol['abstain_costs']):
        adjusted = prob.astype(np.float16) * np.array([1., pc, 1., ac],dtype=np.float16)
        adjusted = (adjusted / adjusted.sum(-1, keepdims=True)).astype(np.float16)
        winner = adjusted.argmax(-1)
        block = (adjusted[None, :, 1] >= pt) & (winner[None, :] == 1)
        refusal = (~block) & (adjusted[None, :, 3] >= at) & (winner[None, :] == 3)
        correct = np.broadcast_to(base, block.shape).copy()
        correct[block] = np.broadcast_to(param_ok, block.shape)[block]
        correct[refusal] = False
        ext_at_risk = ((block | refusal) & (y == 0) & base).sum(-1)
        ext_block = (block & (y == 0)).sum(-1)
        align_at_risk = ((block | refusal) & (y == 2) & base).sum(-1)
        ext = correct[:, y == 0].mean(-1); param = correct[:, y == 1].mean(-1)
        h = np.divide(2*ext*param, ext+param, out=np.zeros_like(ext), where=ext+param > 0)
        pred_ref = np.broadcast_to(base_refusal, block.shape).copy()
        pred_ref[block] = False  # actual blocked refusal is unknown during the CPU screen
        pred_ref[refusal] = True
        tp = (pred_ref & (y == 3)).sum(-1)
        f1 = 2*tp / np.maximum(pred_ref.sum(-1) + (y == 3).sum(), 1)
        align = correct[:, y == 2].mean(-1)
        for j, (p_threshold, a_threshold) in enumerate(thresholds):
            eligible = ext_block[j] <= protocol['max_ext_blocked'] and ext_at_risk[j] <= protocol['max_ext_correct_at_risk'] and align_at_risk[j] <= protocol['max_align_correct_at_risk']
            records.append([pc, ac, p_threshold, a_threshold, float(h[j]), float(f1[j]), float(align[j]),
                int(ext_block[j]), int(ext_at_risk[j]), int(align_at_risk[j]), int(eligible)])
    return np.asarray(records)


def rank_key(record):
    return (record[10], record[4], record[5], record[6], -record[7])


def fit_screen(data, cache, oracle_folder, output, protocol_path='configs/early_gate.json'):
    output=Path(output); output.mkdir(parents=True, exist_ok=True)
    protocol=json.loads((ROOT/protocol_path).read_text())
    seeds=seed_all(protocol['seed'])
    native=read_jsonl(Path(data)/'native.jsonl')
    manifest=json.loads((Path(data)/'feature_manifest.json').read_text())
    assert manifest['ids'] == [r['id'] for r in native]
    cal_index=[i for i,r in enumerate(native) if r['split']=='calibration']
    val_index=[i for i,r in enumerate(native) if r['split']=='val']
    calibration=[native[i] for i in cal_index]; validation=[native[i] for i in val_index]
    assert len(calibration)==704 and len(validation)==242
    for key in ['fact_id','subject_id','question']:
        assert not {r[key] for r in calibration} & {r[key] for r in validation}
    if Path(cache).is_dir():
        export=json.loads((Path(cache)/'export_manifest.json').read_text())
        assert export['ids']==manifest['ids']
        x=np.zeros(manifest['shape'],dtype=np.float16)
        for l in [4,8,10,12,14,16]:
            file=Path(cache)/f'layer{l}.npy'
            assert hashlib.sha256(file.read_bytes()).hexdigest()==export['files'][file.name]
            x[:,l-1]=np.load(file,allow_pickle=False)
    else:x=np.load(cache,mmap_mode='r',allow_pickle=False)
    assert list(x.shape)==manifest['shape'] and x.dtype==np.float16
    # Only these slices are ever materialized. Old test features are never indexed.
    xc=x[cal_index, :16]; xv=x[val_index, :16]
    assert np.isfinite(xc).all() and np.isfinite(xv).all()
    yc=np.array([STATES.index(r['state']) for r in calibration])
    oracle=load_oracle(oracle_folder)
    dump(output/'protocol.json',protocol)
    dump(output/'data_usage.json',{'seed_state':seeds,'calibration_ids':[r['id'] for r in calibration],
        'validation_ids':[r['id'] for r in validation], 'split_id':calibration[0]['split_id'],
        'class_counts':np.bincount(yc).tolist(),'test_features_accessed':0,'test_generations':0,
        'feature_source':str(cache),'feature_manifest_sha256':hashlib.sha256((Path(data)/'feature_manifest.json').read_bytes()).hexdigest(),
        'offline_dtype':'float64; stable SVD, regularized fitting and SPD solves',
        'online_dtype':'float16; folded coefficients and per-layer score accumulation',
        'screen_scope':'conservative cache risk screen; not measured automatic end-to-end performance',
        'unknown_10_14':'intersection of neighboring Oracle successes for screening only; no monotonicity claim'})
    models=[]; grids=[]; shortlisted=[]; failed=[]
    with threadpool_limits(limits=4):
        for layer,mode in itertools.product(protocol['layers'],protocol['feature_modes']):
            layers=source_layers(layer,mode)
            train=features(xc,layers,mode); val=features(xv,layers,mode)
            center=train.mean(0); scale=np.maximum(train.std(0),1e-5)
            z=(train-center)/scale; zv=(val-center)/scale
            pca=PCA(n_components=max(protocol['pca_ranks']),svd_solver='randomized',random_state=protocol['seed']).fit(z)
            zp=pca.transform(z); vp=pca.transform(zv)
            variants=[('pca_logistic',r) for r in protocol['pca_ranks']]+[('full_logistic',0),('shrink_lda',0)]
            for family,r in variants:
                settings=([{'shrinkage':a} for a in protocol['lda_shrinkage']] if family=='shrink_lda' else
                    [{'C':c,'class_weight':w} for c,w in itertools.product(protocol['logistic_c'],protocol['class_weights'])])
                for setting in settings:
                    spec={'layer':layer,'mode':mode,'layers':layers,'family':family,'rank':r,**setting}
                    key=fingerprint(spec)
                    try:
                        if r:
                            whitening=np.sqrt(np.maximum(pca.explained_variance_[:r],1e-10))
                            a=zp[:,:r]/whitening; b=vp[:,:r]/whitening
                        else:a,b=z,zv
                        diag={}
                        if family=='shrink_lda':w,offset,diag=shrink_lda(a,yc,setting['shrinkage'])
                        else:
                            clf=LogisticRegression(C=setting['C'],class_weight=None if setting['class_weight']=='empirical' else 'balanced',
                                max_iter=1500,solver='lbfgs',random_state=protocol['seed'],tol=1e-7)
                            with warnings.catch_warnings():
                                warnings.simplefilter('error',ConvergenceWarning); clf.fit(a,yc)
                            w,offset=clf.coef_,clf.intercept_
                            diag['iterations']=clf.n_iter_.tolist()
                        blocks,bias=fold_map(w,offset,center,scale,layers,mode,
                            pca.components_[:r] if r else None,whitening if r else None)
                        folded=sum(xv[:,l-1].astype(np.float64)@weight for l,weight in zip(layers,blocks))+bias
                        expected=b@w.T+offset
                        error=float(np.max(np.abs(folded-expected)))
                        assert error < 1e-6 and np.isfinite(blocks.astype(np.float16)).all() and np.isfinite(bias.astype(np.float16)).all()
                        # Match the online half accumulation rather than selecting on float64 only.
                        # Emulate addmm's fused accumulation and one fp16 output rounding.
                        raw=np.concatenate([xv[:,l-1] for l in layers],axis=1).astype(np.float64)
                        half=(raw@blocks.astype(np.float16).reshape(-1,4).astype(np.float64)+bias.astype(np.float16)).astype(np.float16)
                        assert np.isfinite(half).all()
                        prob=softmax(half.astype(np.float64),axis=-1)
                        grid=screening_grid(prob,validation,layer,oracle,protocol)
                        best=max(grid,key=rank_key)
                        np.savez_compressed(output/(key+'.npz'),weights=blocks.astype(np.float16),bias=bias.astype(np.float16),
                            layers=layers,validation_logits=half,weights_float64=blocks,bias_float64=bias)
                        item={'key':key,'spec':spec,'screen':best.tolist(),'fold_max_error':error,
                            'half_logit_max_error':float(np.max(np.abs(half.astype(np.float64)-expected))),**diag}
                        models.append(item); grids.append(grid)
                    except (ValueError,AssertionError,np.linalg.LinAlgError,ConvergenceWarning) as exc:
                        failed.append({'spec':spec,'error':str(exc),'type':type(exc).__name__})
            print({'layer':layer,'mode':mode,'fitted':len(models),'failed':len(failed)},flush=True)
    assert models
    # Keep diversity: per layer, top routes must differ, avoiding repeated identical masks.
    for layer in protocol['layers']:
        seen=set()
        for item in sorted([m for m in models if m['spec']['layer']==layer],key=lambda m:rank_key(m['screen']),reverse=True):
            if not item['screen'][10]:continue
            s=item['screen']; policy=dict(zip(['param_cost','abstain_cost','param_threshold','abstain_threshold'],s[:4]))
            saved=np.load(output/(item['key']+'.npz')); prob=softmax(saved['validation_logits'].astype(float),axis=-1)
            route=route_probabilities(prob,policy); sig=route.tobytes()
            if sig in seen:continue
            seen.add(sig)
            selected={**item,'policy':policy,'expected_routes':route.tolist(),
                'artifact_sha256':hashlib.sha256((output/(item['key']+'.npz')).read_bytes()).hexdigest()}
            shortlisted.append(selected)
            if len(seen)>=protocol['top_per_layer']:break
    dump(output/'models.json',models); dump(output/'failed_configurations.json',failed)
    np.savez_compressed(output/'threshold_grid.npz',values=np.stack(grids),model_keys=np.array([m['key'] for m in models]),
        columns=['param_cost','abstain_cost','param_threshold','abstain_threshold','screen_h','screen_refusal_f1',
            'screen_align','ext_blocked','ext_correct_at_risk','align_correct_at_risk','eligible'])
    dump(output/'shortlist.json',{'candidates':shortlisted,'total_fits':len(models),'total_policy_trials':sum(len(g) for g in grids),
        'screen_only':True,'requires_real_generation':True})
    print({'shortlisted':len(shortlisted),'fits':len(models),'failed':len(failed)},flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--data',required=True); p.add_argument('--cache',required=True)
    p.add_argument('--oracle',default='results/oracle_fullmask_v1_cloud/results/oracle_fullmask_v1')
    p.add_argument('--output',default='results/early_gate_cpu_v1')
    a=p.parse_args();fit_screen(a.data,a.cache,a.oracle,a.output)
