# Standard library
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import mujoco as mj
import numpy as np
import numpy.typing as npt

# Local libraries
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder
from ariel.utils.optimizers.revde import RevDE

from initialise import construct_core, save_genotype

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
SPAWN_POS_RUGGED = [1.3, 0, 0.1]

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
    robot_core: Any,
    controller: Controller,
    duration: int = 5,
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
    if terrain == "flat":
        world.spawn(robot_core, position=SPAWN_POS)
    if terrain == "rugged":
        world.spawn(robot_core, position=SPAWN_POS_RUGGED)

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

    simple_runner(
        model,
        data,
        duration=duration,
    )


def calculate_fitness(core, fitness_function, duration=10, terrain="flat"):

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

    experiment_body(robot_core=core, controller=ctrl, duration=duration, terrain="flat")
    mj.set_mjcb_control(None)

    fitness = fitness_function(tracker.history["xpos"][0])

    return fitness


def crossover_and_mutation(robots, pop_size, folder, scaling_factor=-0.5):
    revde = RevDE(scaling_factor)
    # k = 3
    random_choice1 = RNG.choice([1, 2])
    random_choice2 = RNG.choice([i for i in range(3,pop_size)])
    parent_indices = [0, random_choice1, random_choice2]
    RNG.shuffle(parent_indices)
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
        robots[len(robots)] = [genotype, f"{folder}/{save_genotype(genotype, folder)}", construct_core(genotype)]
    return robots


def pick_best_robots(robots, pop_size):
    robots_keys, fitness_scores = [], []
    for key in robots:
        robots_keys.append(key)
        fitness_scores.append(robots[key][-1])

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

    # sorted lists are shortened to return parents and offspring together to
    # original population size
    robots_temp = {}
    for i in keys_sorted[:pop_size]:
        robots_temp[len(robots_temp)] = robots[i]
    
    robots = robots_temp

    return robots
