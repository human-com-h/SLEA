"""Fact-paired, prior-aligned inference with frozen confidence probabilities."""
import numpy as np
from scipy.stats import binomtest
from .common import exact_match,is_refusal
from .calibration import utility_correct

METRICS=['ext_accuracy','param_accuracy','align_accuracy','h_score','refusal_f1','ece','brier']


def sufficient(rows):
    facts=sorted({r['fact_id'] for r in rows});lookup={f:i for i,f in enumerate(facts)}
    x=np.zeros((len(facts),21));known=np.zeros(len(facts),dtype=bool)
    for r in rows:
        i=lookup[r['fact_id']];s=r['state'];correct=exact_match(r['answer'],r['gold_answers'])
        for j,state in enumerate(['ext','param','align']):
            if s==state:x[i,2*j]+=correct;x[i,2*j+1]+=1
        refusal=is_refusal(r['answer']);abstain=s=='abstain'
        x[i,6]+=refusal and abstain;x[i,7]+=refusal;x[i,8]+=abstain
        p=float(r['utility_confidence_frozen']);y=utility_correct(r)
        assert 0<=p<=1
        x[i,9]+=(p-y)**2;x[i,10]+=1
        x[i,11+min(int(p*10),9)]+=y-p
        known[i]=s in ['param','align']
    assert np.all(x[:,10]==2),'Fact clusters must contain both contexts'
    return facts,x,known


def metrics(z):
    z=np.atleast_2d(z)
    ratio=lambda a,b:np.divide(a,b,out=np.zeros_like(a),where=b!=0)
    ext=ratio(z[:,0],z[:,1]);param=ratio(z[:,2],z[:,3]);align=ratio(z[:,4],z[:,5])
    return np.stack([ext,param,align,ratio(2*ext*param,ext+param),ratio(2*z[:,6],z[:,7]+z[:,8]),
        np.sum(np.abs(z[:,11:]),axis=1)/z[:,10],z[:,9]/z[:,10]],axis=-1)


def prior_weights(known,target=.1):
    fraction=np.mean(known);assert 0<fraction<1
    return np.where(known,target/fraction,(1-target)/(1-fraction))


def bootstrap_counts(known,repeats=10000,seed=42):
    rng=np.random.default_rng(seed);counts=np.zeros((repeats,len(known)))
    for state in [True,False]:
        indices=np.flatnonzero(known==state);n=len(indices)
        counts[:,indices]=rng.multinomial(n,np.ones(n)/n,size=repeats)
    return counts


def summarize_with_ci(rows,counts=None,repeats=10000):
    facts,x,known=sufficient(rows)
    if counts is None:counts=bootstrap_counts(known,repeats)
    result={'facts':len(facts),'rows':len(rows),'known_facts':int(known.sum()),'unknown_facts':int((~known).sum())}
    for name,w in [('raw',np.ones(len(facts))),('prior_aligned',prior_weights(known))]:
        point=metrics(np.sum(x*w[:,None],axis=0))[0];boot=metrics(counts@(x*w[:,None]));ci=np.quantile(boot,[.025,.975],axis=0)
        result[name]={metric:{'value':float(point[i]),'ci95':[float(ci[0,i]),float(ci[1,i])]} for i,metric in enumerate(METRICS)}
    return result


def paired_test(a,b,repeats=10000,seed=42):
    fa,xa,known=sufficient(a);fb,xb,kb=sufficient(b);assert fa==fb and np.array_equal(known,kb)
    rng=np.random.default_rng(seed);swap=rng.integers(0,2,size=(repeats,len(fa))).astype(float)
    result={}
    for name,w in [('raw',np.ones(len(fa))),('prior_aligned',prior_weights(known))]:
        aa=xa*w[:,None];bb=xb*w[:,None];sa=aa.sum(0);sb=bb.sum(0)
        observed=(metrics(sa)-metrics(sb))[0]
        mixed_a=sb+swap@(aa-bb);mixed_b=sa+sb-mixed_a
        perm=metrics(mixed_a)-metrics(mixed_b)
        p=(1+np.sum(np.abs(perm)>=np.abs(observed)[None,:]-1e-14,axis=0))/(repeats+1)
        result[name]={metric:{'difference_a_minus_b':float(observed[i]),'p_two_sided':float(p[i])} for i,metric in enumerate(METRICS)}
    by_id={r['id']:r for r in b};mcnemar={}
    for state in ['ext','param','align']:
        discordant=[0,0]
        for row in a:
            if row['state']!=state:continue
            ca=exact_match(row['answer'],row['gold_answers']);cb=exact_match(by_id[row['id']]['answer'],row['gold_answers'])
            discordant[0]+=int(ca and not cb);discordant[1]+=int(cb and not ca)
        n=sum(discordant)
        mcnemar[state]={'a_only_correct':discordant[0],'b_only_correct':discordant[1],
            'p_exact_two_sided':float(binomtest(discordant[0],n,.5).pvalue) if n else 1.}
    result['mcnemar']=mcnemar
    return result


def holm(pvalues):
    order=np.argsort(pvalues);adjusted=np.empty(len(order));running=0.
    for j,i in enumerate(order):running=max(running,(len(order)-j)*pvalues[i]);adjusted[i]=min(running,1.)
    return adjusted.tolist()
