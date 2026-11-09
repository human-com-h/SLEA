"""CPU synthetic check of the deployed gate and original feature collector."""
import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from tests.test_causalgate_stage_a_v1 import CausalGateStageATests
suite=unittest.defaultTestLoader.loadTestsFromTestCase(CausalGateStageATests)
if not unittest.TextTestRunner(verbosity=2).run(suite).wasSuccessful():raise SystemExit(1)
