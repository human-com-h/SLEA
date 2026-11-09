"""Recompute point metrics from text-free per-sample experimental outcomes."""
import csv,json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def summarize(rows):
    rates={s:sum(r['correct'] for r in rows if r['state']==s)/sum(r['state']==s for r in rows) for s in ['ext','param','align','abstain']}
    e,p=rates['ext'],rates['param']
    return dict(ext_accuracy=e,param_accuracy=p,align_accuracy=rates['align'],abstain_refusal=rates['abstain'],h_score=2*e*p/(e+p) if e+p else 0,macro_utility=sum(rates.values())/4)
def main():
    reference=json.loads((ROOT/'results/nqswap_stage_b/analysis.json').read_text())
    output=[]
    for p in sorted((ROOT/'results/outcomes').glob('*/*.jsonl')):
        rows=[json.loads(line) for line in p.read_text().splitlines()]
        metrics=summarize(rows)
        expected=reference['models'][p.parent.name]['points'].get(p.stem)
        if expected:
            for k,v in metrics.items():
                assert abs(v-expected[k]['value'])<1e-12,(p,k,v,expected[k])
        output.append(dict(model=p.parent.name,method=p.stem,n=len(rows),**metrics))
    out=ROOT/'results/rebuilt';out.mkdir(exist_ok=True)
    with (out/'nqswap_confirmation.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    (out/'nqswap_confirmation.json').write_text(json.dumps(output,indent=2))
    print(f'Rebuilt {len(output)} method/model rows from recorded individual outcomes; archived point estimates agree.')
    reference=json.loads((ROOT/'results/reviewer_step_b_results.json').read_text())
    output=[]
    for p in sorted((ROOT/'results/source_outcomes').glob('*/*.jsonl')):
        rows=[json.loads(line) for line in p.read_text().splitlines()];metrics=summarize(rows)
        expected=reference['models'][p.parent.name]['summaries'].get(p.stem)
        if expected:
            expected=expected['macro_4state']
            for k,v in metrics.items():
                key={'abstain_refusal':'abstain_refusal_rate','macro_utility':'overall_utility'}.get(k,k)
                assert abs(v-expected[key]['value'])<1e-12,(p,k)
        output.append(dict(model=p.parent.name,method=p.stem,n=len(rows),**metrics))
    with (out/'source_main.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(output[0]));w.writeheader();w.writerows(output)
    print(f'Rebuilt {len(output)} source-domain rows; archived point estimates agree.')
if __name__=='__main__':main()
