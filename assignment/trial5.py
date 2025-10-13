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
POP_SIZE = 8

INPUT_SIZE = 13 # len(data.qpos) (15) - 3 head global positional args + sinusoidal clock
FIRST_HIDDEN_SIZE = 16 # custom, 'funnel' effect 
SECOND_HIDDEN_SIZE = 12
OUTPUT_SIZE = 8 # controls

HISTORY = []
DURATION = 7

def movement_fitness(history: list[float]) -> float:
    """Check if the spawned body is able to move at all"""

    xs, ys, zs = history[2]
    xc, yc, zc = history[-1]

    distance_from_spawn = np.sqrt((xs - xc) ** 2 + (ys - yc) ** 2)
    return distance_from_spawn

def fitness_function2(history: list[float], terrain="flat"):
    """Rewards positive x-movement and discourages any y-movement"""
    
    xs, ys, zs = history[2]
    xc, yc, zc = history[-1]
    fitness = (xc - xs) - abs(yc - ys)
    return fitness

def fitness_function3(history: list[float]):
    """Rewards positive x-movement."""
    
    xs, ys, zs = history[2]
    xc, yc, zc = history[-1]
    fitness = (xc - xs)
    return fitness

def fitness_function4(history: list[float], terrain="flat") -> float:
    """Rewards any x-movement."""

    xs, ys, zs = history[2]
    xc, yc, zc = history[-1]
    distance_from_spawn = np.sqrt((xs - xc) ** 2)
    return distance_from_spawn

def fitness_function5(history: list[float], terrain="flat", a=0.5) -> float:
    """Rewards y-movement more than x-movement"""
    xs, ys, zs = history[2]
    xc, yc, zc = history[-1]

    distance_from_spawn = np.sqrt(a*(ys - yc) ** 2 + (xs - xc) ** 2)
    return distance_from_spawn


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


def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu

    # Initialize the networks weights randomly
    # Normally, you would use the genes of an individual as the weights,
    # Here we set them randomly for simplicity.
    w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
    w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
    w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))

    # Get inputs, in this case the positions of the actuator motors (hinges)
    inputs = data.qpos

    # Run the inputs through the lays of the network.
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs
    return outputs * np.pi


def experiment_body(
    robot: Any,
    controller: Controller,
    duration: int = 5,
    mode: ViewerTypes = "viewer",
    terrain: str = "flat"
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena()

    # Spawn robot in the world
    # Check docstring for spawn conditions
    if terrain == "rough":
        world.spawn(robot, position=SPAWN_POS_ROUGH)
    if terrain == "tilted":
        world.spawn(robot, position=SPAWN_POS_TILTED)
    if terrain == "flat":
        world.spawn(robot, position=SPAWN_POS)

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            # This disables visualisation (fastest option)
            simple_runner(
                model,
                data,
                duration=duration,
            )
        case "frame":
            # Render a single frame (for debugging)
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
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
    while data.time < duration:
        mujoco.mj_step(model,data)
       
    # Return 0 if history is empty (simulation failed)
    if not HISTORY:
        return 0.0
        
    # fitness is the forward motion, penalty for sideways motion is included
    # tilted world are compensated for 
    fitness = fitness_function6(HISTORY)

    return fitness

def experiment_brain(policy, robot_core_string):
    fitness_flat = experiment_brain_one_terrain(policy, robot_core_string, terrain="flat")
    fitness_rough = experiment_brain_one_terrain(policy, robot_core_string, terrain="rough")
    fitness_tilted = experiment_brain_one_terrain(policy, robot_core_string, terrain="tilted")
    
    fitness = fitness_flat + fitness_rough + fitness_tilted

    return fitness

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

    print(f"[detect_io_sizes] qpos={len(data.qpos)}, nu={model.nu}, input_size={input_size}, output_size={output_size}")
    return input_size, output_size


def evolution_brain(label, robot_core_string, terrain="flat", generations=200):

    input_size, output_size = detect_io_sizes(robot_core_string)
    print(f"Detected input_size={input_size}, output_size={output_size}")

    problem = NEProblem(
        network=lambda: Policy(input_size=input_size, output_size=output_size),
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
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    torch.save(best_policy.state_dict(), f"__gecks__/{label}_best_{timestamp}.pth")
    return best_policy, f"__gecks__/{label}_best_{timestamp}.pth"


def calculate_fitness(core, duration=10, fitness_function=movement_fitness, terrain="flat", mode="simple"):

    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    ctrl = Controller(
        controller_callback_function=nn_controller,
        tracker=tracker,
    )

    experiment_body(robot=core, controller=ctrl, mode=mode, duration=duration, terrain=terrain)
    mj.set_mjcb_control(None)

    fitness = fitness_function(tracker.history["xpos"][0])

    return fitness


def construct_core(genotype):

    mujoco.set_mjcb_control(None)

    p_matrices = NDE.forward(genotype)
    robot_graph: DiGraph[Any] = HPD.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    core = construct_mjspec_from_graph(robot_graph)
    return mujoco.MjSpec.from_string(core.spec.to_xml())


def crossover_and_mutation(robots, scaling_factor=-0.5):
    revde = RevDE(scaling_factor)
    # k = 3
    random_choice1 = RNG.choice([1, 2])
    random_choice2 = RNG.choice([i for i in range(3,POP_SIZE)])
    parent_indices = [0, random_choice1, random_choice2]
    RNG.shuffle(parent_indices)
    print(parent_indices)
    genotype_parents = [robots[key][0] for key in parent_indices]

    offspring_list = []
    for i in range(3):
        split_parents = [g[i] for g in genotype_parents]
        split_offspring = revde.mutate(
            np.array(split_parents[0]),
            np.array(split_parents[1]),
            np.array(split_parents[2])
        )
        for child in split_offspring:
            child = np.clip(child, 0, 1).astype(np.float32)
            offspring_list.append(child)

    # group triplets
    mutated_genotype_offspring = [offspring_list[i:i+3] for i in range(0, len(offspring_list), 3)]

    for genotype in mutated_genotype_offspring:
        core = construct_core(genotype)
        filename = save_genotype(genotype)
        robots[len(robots)] = [genotype, f"trial5/robots/{filename}", core]
    return robots


def pick_best_robots(robots, n=POP_SIZE):
    robots_keys, fitness_scores = [], []
    for key in robots:
        robots_keys.append(key)
        fitness_scores.append(robots[key][5])

    print(fitness_scores)

    fitnesses_sorted, keys_sorted = [fitness_scores[0]], [robots_keys[0]]
    for i in range(1,len(fitness_scores)):
        added = False

        # if value at index i is larger than value at index j, it is added in
        # front of it in sorted list
        for j in range(len(fitnesses_sorted)):
            if fitness_scores[i] >= fitnesses_sorted[j] and added == False:
                fitnesses_sorted.insert(j, fitness_scores[i])
                keys_sorted.insert(j, robots_keys[i])
                added = True

        # if value at index i has not been added to sorted list yet, it has
        # smallest fitness value so is added to the back
        if added == False:
            fitnesses_sorted.append(fitness_scores[i])
            keys_sorted.append(robots_keys[i])
    
    print(keys_sorted[:n])

    # sorted lists are shortened to return parents and offspring together to
    # original population size
    robots_temp = {}
    for i in keys_sorted[:n]:
        robots_temp[len(robots_temp)] = robots[i]
    
    robots = robots_temp

    return robots


def save_genotype(genotype):
    p_matrices = NDE.forward(genotype)

    # Decode the high-probability graph
    robot_graph: DiGraph[Any] = HPD.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )
    # ? ------------------------------------------------------------------ #
    # Save the graph to a file
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"robot_graph_{timestamp}.json"
    save_graph_as_json(
        robot_graph,
        DATA_ROBOTS / filename
    )
    return filename


def initialise_genotype():
    genotype_size = 64
    type_p_genes = RNG.random(genotype_size).astype(np.float32)
    conn_p_genes = RNG.random(genotype_size).astype(np.float32)
    rot_p_genes = RNG.random(genotype_size).astype(np.float32)

    genotype = [
        type_p_genes,
        conn_p_genes,
        rot_p_genes,
    ]

    return genotype


def main(body_gens, brain_gens):
    # order in dictionary: place in population : [genotype, genotype_filename_string, core, best_policy, best_policy_filename, fitness]
    robots = {}

    # initialise robots with enough general movement to population
    while len(robots) < POP_SIZE:
        genotype = initialise_genotype()
        core = construct_core(genotype)
        fitness = calculate_fitness(core,duration=4,fitness_function=fitness_function3)
        if fitness > 0.1:
            robots[len(robots)] = [genotype, f"trial5/robots/{save_genotype(genotype)}", core]

    for _ in range(body_gens):
        # create 3 offspring
        robots = crossover_and_mutation(robots)

        # apply SNES to population and calculate fitness
        for i in range(len(robots)):
            if len(robots[i]) < 4:
                best_policy, best_policy_filename = evolution_brain(f"{robots[i][1]}_brain", robots[i][1], generations=brain_gens)
                robots[i].append(best_policy)
                robots[i].append(best_policy_filename)
                fitness = experiment_brain(robots[i][3], robots[i][1])
                robots[i].append(fitness)

        # pick 5 best robots
        robots = pick_best_robots(robots)

        print(robots[0])


if __name__ == "__main__":
    main(body_gens=40, brain_gens=20)
