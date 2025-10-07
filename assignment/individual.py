# Third-party libraries
import numpy as np
import mujoco as mj
from mujoco import viewer
import matplotlib.pyplot as plt

from ariel.simulation.environments import OlympicArena
from ariel.simulation.controllers.controller import Controller
from ariel.utils.renderers import single_frame_renderer, video_renderer

from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json,
)
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.utils.tracker import Tracker

from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder

from pathlib import Path
DATA = Path("./__data__")
DATA.mkdir(exist_ok=True)

import random, numpy as np
import torch 

seed = 9
def seed_all(s=seed):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

#from config import DURATION, LAT_LAMBDA,N,pos,TARGET_POSITION
duration= 10 # 10s sim
Lat_lambda = 1 # lateral movement penalty
INPUT_SIZE = 13 # len(data.qpos) (15) - 3 head global positional args + sinusoidal clock
FIRST_HIDDEN_SIZE = 16 # custom, 'funnel' effect 
SECOND_HIDDEN_SIZE = 12
OUTPUT_SIZE = 8
pos = [0.0, 0, 1.0]
target_pos= [5, 0, 0.5] 
N =20
useless_body = -100000
min_dis = 0.1

def nn_controller(model: mj.MjModel, data: mj.MjData, weights):
    w1, w2, w3 = weights
    clock = np.sin(2 * data.time)
    

    need = w1.shape[0] - 1          # how many qpos the net expects (minus the clock)
    inputs = np.concatenate([data.qpos[-need:], [clock]])
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    return outputs * np.pi


"""
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

    """

# I kept this function the same as in template only added check for pixel_dis is 0 

def show_xpos_history(history: list[float]) -> None:
    # Create a tracking camera
    camera = mj.MjvCamera()
    camera.type = mj.mjtCamera.mjCAMERA_FREE
    camera.lookat = [2.5, 0, 0]
    camera.distance = 10
    camera.azimuth = 0
    camera.elevation = -90

    # Initialize world to get the background
    mj.set_mjcb_control(None)
    world = OlympicArena()
    model = world.spec.compile()
    data = mj.MjData(model)
    save_path = str(DATA / "background.png")
    single_frame_renderer(
        model,
        data,
        camera=camera,
        save_path=save_path,
        save=True,
    )

    # Setup background image
    img = plt.imread(save_path)
    _, ax = plt.subplots()
    ax.imshow(img)
    w, h, _ = img.shape

    # Convert list of [x,y,z] positions to numpy array
    pos_data = np.array(history)

    # Calculate initial position
    x0, y0 = int(h * 0.483), int(w * 0.815)
    xc, yc = int(h * 0.483), int(w * 0.9205)
    ym0, ymc = 0, pos[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))

    # Avoide deviding by zero
    if pixel_to_dist == 0:
        pixel_to_dist = 1e-5

    pos_data_pixel = [[xc, yc]]
    for i in range(len(pos_data) - 1):
        xi, yi, _ = pos_data[i]
        xj, yj, _ = pos_data[i + 1]
        xd, yd = (xj - xi) / pixel_to_dist, (yj - yi) / pixel_to_dist
        xn, yn = pos_data_pixel[i]
        pos_data_pixel.append([xn + int(xd), yn + int(yd)])
    pos_data_pixel = np.array(pos_data_pixel)

    # Plot x,y trajectory
    ax.plot(x0, y0, "kx", label="[0, 0, 0]")
    ax.plot(xc, yc, "go", label="Start")
    ax.plot(pos_data_pixel[:, 0], pos_data_pixel[:, 1], "b-", label="Path")
    ax.plot(pos_data_pixel[-1, 0], pos_data_pixel[-1, 1], "ro", label="End")

    # Add labels and title
    ax.set_xlabel("X Position")
    ax.set_ylabel("Y Position")
    ax.legend()

    # Title
    plt.title("Robot Path in XY Plane")

    # Show results
    plt.show()

def run_individual(body_genes, weights=None, model=None,robot_spec=None, view=False, video=False):
    
    mj.set_mjcb_control(None)
    world = OlympicArena()

    seed_all()
    nde = NeuralDevelopmentalEncoding(number_of_modules=N)
    p_matrices = nde.forward(body_genes)
    hpd = HighProbabilityDecoder(N)
    robot_graph = hpd.probability_matrices_to_graph(*p_matrices)
    robot = construct_mjspec_from_graph(robot_graph)

    world.spawn(robot.spec, spawn_position=pos)
    model = world.spec.compile()
    data = mj.MjData(model)

    # Stop early if body has no limps
    if model.nu == 0:
        return useless_body
    
    if weights is None:
        input_size = len(data.qpos) + 1 
        hidden_size = 8
        output_size = model.nu
        weights = (
            np.random.randn(input_size, hidden_size),
            np.random.randn(hidden_size, hidden_size),
            np.random.randn(hidden_size, output_size),
        )

    # Reset state and time of simulation
    tracker = Tracker(
        mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM,
        name_to_bind="core",
    )
    tracker.setup(world.spec, data)

    ctrl = Controller(
        controller_callback_function=lambda m, d: nn_controller(m, d, weights),
        tracker=tracker,
    )
    mj.set_mjcb_control(None) 
    mj.mj_resetData(model, data)
    mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
    

    while data.time < duration:
        mj.mj_step(model, data)


    # Return 0 if history is empty (simulation failed)
    #if len(tracker.history.get("xpos", [])) == 0:
    #    return -1e6
    if len(tracker.history["xpos"]) == 0 or len(tracker.history["xpos"][0]) == 0:
        return useless_body
    

    #x0,_,_= tracker.history["xpos"][0][0]
    xc, yc, zc = tracker.history["xpos"][0][-1]
    xt, yt, zt = target_pos

    #if (xc - x0) <min_dis:
    #    return useless_body

    # negative Euclidean distance to target
    cartesian_distance = np.sqrt(
        (xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2
    )
    fitness = -cartesian_distance

    if view:
        print(f'ypos_final: {tracker.history['xpos'][0][-1]}')
        print(f'fitness_final: {fitness}')
        viewer.launch(model=model, data=data)
        if len(tracker.history["xpos"][0]) > 0 :
            show_xpos_history(tracker.history["xpos"][0])

    if video:
        rec = VideoRecorder(output_folder="./__videos__")
        video_renderer(model, data, duration=duration, video_recorder=rec)
    return fitness
