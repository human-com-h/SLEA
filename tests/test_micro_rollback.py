import copy,unittest
from types import SimpleNamespace
import torch
from transformers import Qwen2Config,Qwen2ForCausalLM
from src.causalgate_micro_rollback import MicroRollback
from src.phase4_runtime import ScopeController

class Collector:
    def reset(self,span):self.active=True

class MicroRollbackTests(unittest.TestCase):
    def test_replay_matches_direct_control_and_preserves_cache(self):
        torch.manual_seed(42)
        config=Qwen2Config(vocab_size=32,hidden_size=16,intermediate_size=32,num_hidden_layers=4,num_attention_heads=2,num_key_value_heads=2)
        config._attn_implementation='eager'
        model=Qwen2ForCausalLM(config).half().eval()
        reference=copy.deepcopy(model)
        model._phase4_controllers=[];reference._phase4_controllers=[]
        replay=MicroRollback(SimpleNamespace(model=model),1,3,Collector(),lambda _:True)
        direct=ScopeController(SimpleNamespace(model=reference),1,force_route=1)
        ids=torch.tensor([[1,4,5,6,7]])
        try:
            replay.reset([2,3]);direct.reset([2,3])
            with torch.inference_mode():
                actual=model(input_ids=ids,use_cache=True)
                expected=reference(input_ids=ids,use_cache=True)
            self.assertTrue(torch.equal(actual.logits,expected.logits))
            self.assertEqual(replay.discarded_layers,2)
            self.assertEqual(replay.calls_by_layer,[1,2,2,1])
            self.assertEqual(replay.cache_steps,[[5,5,5,5]])
            with torch.inference_mode():
                a=model(input_ids=torch.tensor([[8]]),past_key_values=actual.past_key_values,use_cache=True)
                b=reference(input_ids=torch.tensor([[8]]),past_key_values=expected.past_key_values,use_cache=True)
            self.assertTrue(torch.equal(a.logits,b.logits))
            self.assertEqual(replay.cache_steps[-1],[6,6,6,6])
        finally:replay.close();direct.close()

if __name__=='__main__':unittest.main()
