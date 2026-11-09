import unittest
from unittest.mock import patch
from types import SimpleNamespace
import numpy as np
import torch
from sklearn.covariance import ledoit_wolf_shrinkage
from src.causalgate_learning_v1 import lw_dual_weight,threshold_pair,decide
from src.causalgate_features_v1 import CausalFeatures


class Attn(torch.nn.Module):
    def forward(self,x,output_attentions=False):
        n=x.shape[1];weights=torch.ones((1,2,n,n),dtype=torch.float16)/n
        return x,weights if output_attentions else None,None


class Layer(torch.nn.Module):
    def __init__(self):super().__init__();self.self_attn=Attn();self.shift=0.
    def forward(self,x):return (self.self_attn(x,output_attentions=False)[0]+self.shift,)


class Toy(torch.nn.Module):
    def __init__(self):
        super().__init__();self.model=torch.nn.Module();self.model.layers=torch.nn.ModuleList([Layer() for _ in range(8)])
    def forward(self,input_ids):
        x=torch.arange(input_ids.shape[1]*8,dtype=torch.float16).reshape(1,-1,8)/16
        for layer in self.model.layers:x=layer(x)[0]
        return x


class CausalGateStageATests(unittest.TestCase):
    def test_dual_ledoit_wolf_matches_covariance_reference(self):
        rng=np.random.default_rng(42);x=rng.normal(size=(30,8));y=np.arange(30)%2
        x[y==1]+=.4
        w,b,alpha=lw_dual_weight(x,y)
        means=np.stack([x[y==k].mean(0) for k in [0,1]]);res=x-means[y]
        expected=ledoit_wolf_shrinkage(res,assume_centered=True)
        self.assertAlmostEqual(alpha,expected,places=12)
        cov=(1-alpha)*(res.T@res/len(res))+alpha*np.mean(res**2)*np.eye(x.shape[1])
        np.testing.assert_allclose(w,np.linalg.solve(cov,means[1]-means[0]),atol=1e-10)

    def test_and_policy_is_rounded_like_deployment_and_ext_guarded(self):
        rows=[dict(state=s) for s in ['ext','abstain','align','param']]
        k=np.array([0.,0.,1.,1.]);c=np.array([0.,1.,0.,1.])
        policy,_=threshold_pair(k,c,rows);mask=decide(k,c,policy)
        self.assertEqual(mask.tolist(),[False,False,False,True])
        self.assertEqual(policy['ext_false_block_rate'],0.)

    def test_future_decoder_changes_cannot_change_prefix_features(self):
        model=Toy();capture=CausalFeatures(model,[6])
        def fake_lens(self,layer,vectors,alpha):
            self.trace.append(['early_lens',layer]);return vectors.mean(-1).repeat_interleave(6)
        with patch.object(CausalFeatures,'early_lens',fake_lens):
            capture.reset([1,4]);model(input_ids=torch.arange(6)[None]);first=capture.vector(6,'all_causal').clone()
            model.model.layers[6].shift=100.;model.model.layers[7].shift=-50.
            capture.reset([1,4]);model(input_ids=torch.arange(6)[None]);second=capture.vector(6,'all_causal').clone()
        self.assertTrue(torch.equal(first,second))
        self.assertTrue(all(entry[1]<=6 for entry in capture.trace))
        capture.close()


if __name__=='__main__':unittest.main()
