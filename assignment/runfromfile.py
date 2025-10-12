# the competition tester doesn't work with our neurocontroller, but this
# file basically does the same thing

import datetime

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import matplotlib.pyplot as plt
import mujoco as mj
import mujoco
import numpy as np
import numpy.typing as npt
from mujoco import viewer

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
    draw_graph
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.optimizers.revde import RevDE

import json
import networkx as nx
import torch
import ray
import os

import torch.nn as nn
import math
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import XNES
from evotorch.logging import StdOutLogger, PandasLogger
from functools import partial

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
type ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# --- DATA SETUP ---
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

DATA_ROBOTS = CWD / "__data__" / SCRIPT_NAME / "robots"
DATA_ROBOTS.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

SPAWN_POS_ROUGH = [1.3, 0, 0.1]
SPAWN_POS_TILTED = [3.0, 0, 0.3]

NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
HPD = HighProbabilityDecoder(NUM_OF_MODULES)
POP_SIZE = 5

INPUT_SIZE = 16 # len(data.qpos) (15) - 3 head global positional args + sinusoidal clock
FIRST_HIDDEN_SIZE = 16 # custom, 'funnel' effect 
SECOND_HIDDEN_SIZE = 12
OUTPUT_SIZE = 11 # controls

HISTORY = []
DURATION = 40

def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (yt - yc) ** 2 + (xt - xc) ** 2 + (zt - zc) ** 2
    )
    return -cartesian_distance

def neuro_controller(model, data, to_track, policy) -> None:
    # sinusclock
    clock = np.sin(2*data.time)

    # input is positions of the actuator motors plus the sinusclock
    inputs = np.concatenate([data.qpos[3:], [clock]])

    # convert inputs to PyTorch tensor
    inputs_tensor = torch.FloatTensor(inputs)

    # Get outputs from the PyTorch policy
    with torch.no_grad():
        outputs = policy(inputs_tensor).numpy()

    data.ctrl = np.clip(outputs, -np.pi/2, np.pi/2)
    # track hist
    HISTORY.append(to_track[0].xpos.copy())

def experiment_brain_one_terrain(policy, robot_core_string, terrain):
    global HISTORY
    HISTORY = []

    mujoco.set_mjcb_control(None)
    
    world = OlympicArena()
    
    with open(robot_core_string) as f:
        data = json.load(f)
    if "edges" in data:
        data["links"] = data.pop("edges")
    robot_graph = nx.node_link_graph(data, edges="links")
    robot_core = construct_mjspec_from_graph(robot_graph)

    if terrain == "rough":
        world.spawn(robot_core.spec, position=SPAWN_POS_ROUGH)
    if terrain == "tilted":
        world.spawn(robot_core.spec, position=SPAWN_POS_TILTED)
    if terrain == "flat":
        world.spawn(robot_core.spec, position=SPAWN_POS)

    model = world.spec.compile()
    data = mujoco.MjData(model) 

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    mujoco.set_mjcb_control(lambda m,d: neuro_controller(m, d, to_track, policy))

    duration = DURATION
    # This records a video of the simulation
    path_to_video_folder = str(DATA / "videos")
    video_recorder = VideoRecorder(output_folder=path_to_video_folder)

    # Render with video recorder
    video_renderer(
        model,
        data,
        duration=duration,
        video_recorder=video_recorder,
    )
       
    # Return 0 if history is empty (simulation failed)
    if not HISTORY:
        return 0.0
    
    print(fitness_function(HISTORY))


class Policy(nn.Module):
    def __init__(self, input_size=INPUT_SIZE,
                 first_hidden_size=FIRST_HIDDEN_SIZE,
                 second_hidden_size=SECOND_HIDDEN_SIZE,
                 output_size=OUTPUT_SIZE):
        super().__init__()

        if input_size is None:
            input_size = 13  # fallback default

        self.net = nn.Sequential(
            nn.Linear(input_size, first_hidden_size),
            nn.Tanh(),
            nn.Linear(first_hidden_size, second_hidden_size),
            nn.Tanh(),
            nn.Linear(second_hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)
        return self.net(x) * (math.pi / 2)

robot_core_string = "usain_ro-bolt.json"
robot_brain_string = "usain_ro-bolt.pth"

showbrain = Policy()
showbrain.load_state_dict(torch.load(robot_brain_string))
experiment_brain_one_terrain(showbrain, robot_core_string, terrain="flat")