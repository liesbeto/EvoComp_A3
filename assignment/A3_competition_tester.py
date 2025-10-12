"""Robot Olympics Competition template code."""

# Standard library
from pathlib import Path
from typing import Any

import mujoco as mj
import numpy as np
import torch
import torch.nn as nn
import math

# Local libraries
from ariel import console
from ariel.body_phenotypes.robogen_lite.constructor import (
    construct_mjspec_from_graph,
)
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    load_graph_from_json,
)
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import video_renderer
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# --- DATA SETUP --- #
SCRIPT_NAME = __file__.split("/")[-1][:-3]
CWD = Path.cwd()
DATA = CWD / "__data__"
DATA.mkdir(exist_ok=True)

# Global variables
SPAWN_POS = [-0.8, 0, 0.1]
NUM_OF_MODULES = 30
TARGET_POSITION = [5, 0, 0.5]
TIME_OUT = 40

# Local scripts
from A3_plot_function import show_xpos_history


def fitness_function(history: list[tuple[float, float, float]]) -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2,
    )
    return -cartesian_distance


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = TIME_OUT,
) -> None:
    """Run the simulation with custom controller."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    # Import environments from ariel.simulation.environments
    world = OlympicArena(
        load_precompiled=False,
    )

    # Spawn robot in the world
    # Check docstring for spawn conditions
    world.spawn(
        robot.spec,
        position=SPAWN_POS,
        correct_collision_with_floor=True,
    )

    # Generate the model and data
    # These are standard parts of the simulation USE THEM AS IS, DO NOT CHANGE
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    controller.tracker.setup(world.spec, data)

    # Set the control callback function
    # This is called every time step to get the next action.
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),  # pyright: ignore[reportUnknownLambdaType]
    )

    # ------------------------------------------------------------------ #

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
    # ==================================================================== #

class Policy(nn.Module):
    
    def __init__(self, input_size=16,
        first_hidden_size=16,
        second_hidden_size=12,
        output_size=11):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_size, first_hidden_size),
            nn.Tanh(),
            nn.Linear(first_hidden_size,second_hidden_size),
            nn.Tanh(),
            nn.Linear(second_hidden_size, output_size),
            nn.Tanh()
        )

    def forward(self, x, hehe):
        return self.net(x) * (math.pi / 2)


def main() -> None:
    """Entry point."""
    # Load your robot graph here
    path_to_graph = CWD / "trial5/robots/marinde1.json"
    robot_graph = load_graph_from_json(path_to_graph)

    # Construct the robot from the graph
    core = construct_mjspec_from_graph(robot_graph)

    mujoco_type_to_find = mj.mjtObj.mjOBJ_GEOM
    name_to_bind = "core"
    tracker = Tracker(
        mujoco_obj_to_find=mujoco_type_to_find,
        name_to_bind=name_to_bind,
    )

    showbrain = Policy()
    showbrain.load_state_dict(torch.load("__gecks__/marinde1_best.pth"))

    # Simulate the robot
    ctrl = Controller(
        controller_callback_function=showbrain,
        tracker=tracker,
    )

    experiment(robot=core, controller=ctrl)

    show_xpos_history(tracker.history["xpos"][0])

    fitness = fitness_function(tracker.history["xpos"][0])
    msg = f"Fitness of generated robot: {fitness}"
    console.log(msg)


if __name__ == "__main__":
    main()