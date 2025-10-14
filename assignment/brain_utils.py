import datetime
from pathlib import Path
from typing import TYPE_CHECKING
import mujoco as mj
import numpy as np
import json
import networkx as nx
import torch
import ray
import torch.nn as nn
import math
from evotorch.neuroevolution import NEProblem
from evotorch.algorithms import XNES
from evotorch.logging import StdOutLogger
from functools import partial

# Local libraries from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder

from fitness_functions import given_fitness_function, fitness_function_complex
from A3_plot_function_copy import show_xpos_history

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph


CWD = Path.cwd()

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
SPAWN_POS_RUGGED = [1.3, 0, 0.1]
SPAWN_POS_TILTED = [3.0, 0, 0.3]

FIRST_HIDDEN_SIZE = 16
SECOND_HIDDEN_SIZE = 12

HISTORY = []


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


def experiment_brain_one_terrain(policy, graph_filename, duration=10, terrain="flat", view=False, folder=None):
    global HISTORY
    HISTORY = []

    mj.set_mjcb_control(None)
    
    world = OlympicArena()
    
    with open(graph_filename) as f:
        data = json.load(f)
    if "edges" in data:
        data["links"] = data.pop("edges")
    robot_graph = nx.node_link_graph(data, edges="links")
    robot_core = construct_mjspec_from_graph(robot_graph)

    if terrain == "rugged":
        world.spawn(robot_core.spec, position=SPAWN_POS_RUGGED)
    if terrain == "tilted":
        world.spawn(robot_core.spec, position=SPAWN_POS_TILTED)
    if terrain == "flat":
        world.spawn(robot_core.spec, position=SPAWN_POS)

    model = world.spec.compile()
    data = mj.MjData(model) 

    geoms = world.spec.worldbody.find_all(mj.mjtObj.mjOBJ_GEOM)
    to_track = [data.bind(geom) for geom in geoms if "core" in geom.name]

    mj.set_mjcb_control(lambda m,d: neuro_controller(m, d, to_track, policy))

    if view == False:
        while data.time < duration:
            mj.mj_step(model,data)

    else:
        # This records a video of the simulation
        video_recorder = VideoRecorder(output_folder=folder)

        # Render with video recorder
        video_renderer(
            model,
            data,
            duration=duration,
            video_recorder=video_recorder,
        )

        history = []
        for i in range(0, len(HISTORY), int(len(HISTORY)/duration)):
            history.append(HISTORY[i])
        history.append(HISTORY[-1])

        show_xpos_history(history)

        print(given_fitness_function(HISTORY))

    # Return 0 if history is empty (simulation failed)
    if not HISTORY:
        return 0.0
        
    # fitness is the forward motion, penalty for sideways motion is included
    # tilted world are compensated for 
    fitness = fitness_function_complex(HISTORY)

    return fitness


def experiment_brain(policy, graph_filename):
    fitness_flat = experiment_brain_one_terrain(policy, graph_filename, terrain="flat")
    fitness_rugged = experiment_brain_one_terrain(policy, graph_filename, terrain="rugged")
    
    fitness = fitness_flat + fitness_rugged

    return fitness


class Policy(nn.Module):
    def __init__(self, input_size, output_size,
                 first_hidden_size=FIRST_HIDDEN_SIZE,
                 second_hidden_size=SECOND_HIDDEN_SIZE):
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


def detect_io_sizes(graph_filename):

    # Load robot graph
    with open(graph_filename) as f:
        data = json.load(f)
    if "edges" in data:
        data["links"] = data.pop("edges")
    robot_graph = nx.node_link_graph(data, edges="links")
    robot_core = construct_mjspec_from_graph(robot_graph)

    mj.set_mjcb_control(None)

    # Spawn robot inside the *actual* world environment
    world = OlympicArena()
    world.spawn(robot_core.spec, position=SPAWN_POS)
    model = world.spec.compile()
    data = mj.MjData(model)

    # Now detect the true sizes
    input_size = len(data.qpos[3:]) + 1  # exclude global xyz, add clock
    output_size = model.nu  # actuators

    return input_size, output_size


def evolution_brain(label, graph_filename, folder, generations=100):

    input_size, output_size = detect_io_sizes(graph_filename)

    problem = NEProblem(
        network=lambda: Policy(input_size, output_size),
        network_eval_func=partial(experiment_brain, graph_filename=graph_filename),
        objective_sense="max",
        num_actors=6
    )

    searcher = XNES(problem, stdev_init=0.9)
    # init logging 
    _ = StdOutLogger(searcher)

    # running
    searcher.run(generations)

    folder_brains = CWD / folder / "brains"
    folder_brains.mkdir(exist_ok=True)

    best_solution = searcher.status['best']
    best_policy = problem.make_net(best_solution)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    torch.save(best_policy.state_dict(), f"{folder_brains}/{label}.pth")
    return best_policy, f"__gecks__/{label}_best_{timestamp}.pth"
