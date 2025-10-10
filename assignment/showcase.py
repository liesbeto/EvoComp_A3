import os 
import sys

parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

import torch
from individual import run_individual
from evolve import GeckoPolicy

showbrain = GeckoPolicy() 
showbrain.load_state_dict(torch.load('./__gecks__/trialtrial2_best.pth'))
run_individual(showbrain, view=False, video=True)
