import datetime

# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import json
import networkx as nx
from networkx.readwrite import json_graph

import torch
import ray
import os

import torch.nn as nn
import math
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import XNES
from evotorch.logging import StdOutLogger, PandasLogger

import matplotlib.pyplot as plt
import mujoco as mj
import mujoco
import numpy as np
import numpy.typing as npt
from mujoco import viewer

from functools import partial

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

SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__" / SCRIPT_NAME
DATA.mkdir(exist_ok=True)

SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]

# ? still need to figure out these locations :/
SPAWN_POS_ROUGH = [1.3, 0, 0.1]
SPAWN_POS_TILTED = [3.5, 0, 0.2]

INPUT_SIZE = 13 # len(data.qpos) (15) - 3 head global positional args + sinusoidal clock
FIRST_HIDDEN_SIZE = 16 # custom, 'funnel' effect 
SECOND_HIDDEN_SIZE = 12
OUTPUT_SIZE = 8 # controls

HISTORY = []
DURATION = 40

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

def random_move(model, data, to_track) -> None:
    num_joints = model.nu
    hinge_range = np.pi/2
    rand_moves = np.random.uniform(low= -hinge_range,
                                   high=hinge_range,
                                   size=num_joints) 
    delta = 0.05
    data.ctrl += rand_moves * delta 
    data.ctrl = np.clip(data.ctrl, -np.pi/2, np.pi/2)
    HISTORY.append(to_track[0].xpos.copy())

def show_qpos_history(history:list):
    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)
    
    # Create figure and axis
    plt.figure(figsize=(10, 6))
    
    # Plot x,y trajectory
    plt.plot(pos_data[:, 0], pos_data[:, 1], 'b-', label='Path')
    plt.plot(pos_data[0, 0], pos_data[0, 1], 'go', label='Start')
    plt.plot(pos_data[-1, 0], pos_data[-1, 1], 'ro', label='End')
    
    # Add labels and title
    plt.xlabel('X Position')
    plt.ylabel('Y Position') 
    plt.title('Robot Path in XY Plane')
    plt.legend()
    plt.grid(True)
    
    # Set equal aspect ratio and center at (0,0)
    plt.axis('equal')
    max_range = max(abs(pos_data).max(), 0.3)  # At least 1.0 to avoid empty plots
    plt.xlim(-max_range, max_range)
    plt.ylim(-max_range, max_range)
    
    plt.show()

def fitness_function6(history=HISTORY, a=0.5) -> float:
    """Rewards positive x-movement and punishes any y-movement"
    (throughout whole history)."""

    xs, ys, zs = history[2]
    xc, yc, zc = history[-1]

    if zc < 0:
        return -10

    average_ydeviation = np.mean(history[:][1])
    fitness = (xc - xs) - a*abs(average_ydeviation)

    return fitness

def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )

    return -cartesian_distance

def experiment_brain(policy, robot_core_string):
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

    world.spawn(robot_core.spec, position=[0, 0, 0])
    
    model = world.spec.compile()
    data = mujoco.MjData(model) 

    geoms = world.spec.worldbody.find_all(mujoco.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    mujoco.set_mjcb_control(lambda m,d: neuro_controller(m, d, to_track, policy))

    duration = DURATION
    while data.time < duration:
        mujoco.mj_step(model,data)
       
    # Return 0 if history is empty (simulation failed)
    if not HISTORY:
        return 0.0
        
    # fitness is the forward motion, penalty for sideways motion is included
    # tilted world are compensated for 
    fitness = fitness_function(HISTORY)

    return fitness

class Policy(nn.Module):
    
    def __init__(self, input_size=INPUT_SIZE, \
        first_hidden_size=FIRST_HIDDEN_SIZE, \
        second_hidden_size=SECOND_HIDDEN_SIZE, \
        output_size=OUTPUT_SIZE):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, first_hidden_size),
            nn.Tanh(),
            nn.Linear(first_hidden_size,second_hidden_size),
            nn.Tanh(),
            nn.Linear(second_hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x):
        return self.net(x) * (math.pi / 2)


def detect_io_sizes(robot_core_string):

    # Load robot graph
    with open(robot_core_string) as f:
        data = json.load(f)
    if "edges" in data:
        data["links"] = data.pop("edges")
    robot_graph = nx.node_link_graph(data, edges="links")
    robot_core = construct_mjspec_from_graph(robot_graph)

    mujoco.set_mjcb_control(None)

    # Spawn robot inside the *actual* world environment
    world = OlympicArena()
    world.spawn(robot_core.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data = mujoco.MjData(model)

    # Now detect the true sizes
    input_size = len(data.qpos[3:]) + 1  # exclude global xyz, add clock
    output_size = model.nu  # actuators

    return input_size, output_size


def evolution_brain(label, robot_core_string, generations=200):

    input_size, output_size = detect_io_sizes(robot_core_string)

    problem = NEProblem(
        network=lambda: Policy(input_size=16, output_size=11),
        network_eval_func=partial(experiment_brain, robot_core_string=robot_core_string),
        objective_sense="max",
        num_actors=6
    )

    searcher = XNES(problem, stdev_init=0.9)
    # init logging 
    _ = StdOutLogger(searcher)
    pandas_logger = PandasLogger(searcher)

    # running
    searcher.run(generations)

    # saving
    if not os.path.exists('__logs__'):
        os.mkdir('__logs__')
        
    df = pandas_logger.to_dataframe()
    df.to_csv(f'__logs__/{label}.csv')
    
    if not os.path.exists('__gecks__'):
        os.mkdir('__gecks__')

    best_solution = searcher.status['best']
    best_policy = problem.make_net(best_solution)
    print(best_policy)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    torch.save(best_policy.state_dict(), f"{label}_best.pth")


def main():

    evolution_brain('usain_ro-bolt_better', "usain_ro-bolt.json", generations=100)
    ray.shutdown()

main()