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

# ? still need to figure out these locations :/
SPAWN_POS_ROUGH = [1.3, 0, 0.1]
SPAWN_POS_TILTED = [3.5, 0, 0.2]

NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
HPD = HighProbabilityDecoder(NUM_OF_MODULES)
POP_SIZE = 5

def movement_fitness(history: list[float], terrain="flat") -> float:
    """Check if the spawned body is able to move at all"""
    xs, ys, zs = SPAWN_POS
    xc, yc, zc = history[-1]

    distance_from_spawn = np.sqrt((xs - xc) ** 2 + (ys - yc) ** 2)
    return distance_from_spawn

def given_fitness_function(history: list[float], terrain="flat") -> float:
    xt, yt, zt = TARGET_POSITION
    xc, yc, zc = history[-1]

    # Minimize the distance --> maximize the negative distance
    cartesian_distance = np.sqrt((xt - xc) ** 2 + (yt - yc) ** 2 + (zt - zc) ** 2)
    return -cartesian_distance

def fitness_function2(history: list[float], terrain="flat"):
    """Rewards positive y-movement and discourages any x-movement"""

    if terrain == "rough":
        spawn_pos = SPAWN_POS_ROUGH
    if terrain == "tilted":
        spawn_pos = SPAWN_POS_TILTED
    if terrain == "flat":
        spawn_pos = SPAWN_POS
    
    xs, ys, zs = spawn_pos
    xc, yc, zc = history[-1]
    fitness = -(yc - ys) - abs(xc - xs)
    return fitness

def fitness_function3(history: list[float], terrain="flat"):
    """Rewards positive y-movement."""

    if terrain == "rough":
        spawn_pos = SPAWN_POS_ROUGH
    if terrain == "tilted":
        spawn_pos = SPAWN_POS_TILTED
    if terrain == "flat":
        spawn_pos = SPAWN_POS
    
    xs, ys, zs = spawn_pos
    xc, yc, zc = history[-1]
    fitness = -(yc - ys)
    return fitness

def fitness_function4(history: list[float], terrain="flat") -> float:
    """Rewards any y-movement."""

    xs, ys, zs = SPAWN_POS
    xc, yc, zc = history[-1]
    distance_from_spawn = np.sqrt((ys - yc) ** 2)
    return distance_from_spawn

def fitness_function5(history: list[float], terrain="flat", a=0.5) -> float:
    """Rewards y-movement more than x-movement"""
    xs, ys, zs = SPAWN_POS
    xc, yc, zc = history[-1]

    distance_from_spawn = np.sqrt(a*(xs - xc) ** 2 + (ys - yc) ** 2)
    return distance_from_spawn


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

def show_xpos_history(history: list[float], terrain="flat") -> None:
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
    if terrain == "rough":
        ym0, ymc = 0, SPAWN_POS_ROUGH[0]
    if terrain == "tilted":
        ym0, ymc = 0, SPAWN_POS_TILTED[0]
    if terrain == "flat":
        ym0, ymc = 0, SPAWN_POS[0]

    # Convert position data to pixel coordinates
    pixel_to_dist = -((ymc - ym0) / (yc - y0))
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
    # plt.show()
    plt.close()


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
        world.spawn(robot, spawn_position=SPAWN_POS_ROUGH)
    if terrain == "tilted":
        world.spawn(robot, spawn_position=SPAWN_POS_TILTED)
    if terrain == "flat":
        world.spawn(robot, spawn_position=SPAWN_POS)

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


def calculate_fitness(core, duration=10, fitness_function=fitness_function2, terrain="flat", mode="simple"):

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

    # i get an error if i comment this function out for some reason but i thought it just makes the plot :/
    show_xpos_history(tracker.history["xpos"][0], terrain=terrain)

    fitness = fitness_function(tracker.history["xpos"][0], terrain=terrain)

    return fitness


def construct_core(genotype):
    p_matrices = NDE.forward(genotype)

    # Decode the high-probability graph
    robot_graph: DiGraph[Any] = HPD.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    core = construct_mjspec_from_graph(robot_graph)
    return mujoco.MjSpec.from_string(core.spec.to_xml())


def initialise_cores(genotypes):
    cores = []
    for genotype in genotypes:
        cores.append(construct_core(genotype))
    return cores


# def crossover_genotypes(genotype_parents, p_crossover=0.1):
#     """Need to crossover the same of the 3 arrays in a genotype."""
    
#     num_of_dims = len(genotype_parents[0][0])

#     genotype_offspring = []
#     for i in range(0, len(genotype_parents), 2):
#         if RNG.random() <= p_crossover:
#             offspring1, offspring2 = [], []
#             for j in range(0, 3):
#                 crossover_point = RNG.randint(1, num_of_dims-1)

#                 offspring1_array = genotype_parents[i][j][:crossover_point] + genotype_parents[i+1][j][crossover_point:]
#                 offspring2_array = genotype_parents[i+1][j][:crossover_point] + genotype_parents[i][j][crossover_point:]
#                 offspring1.append(offspring1_array)
#                 offspring2.append(offspring2_array)
#             genotype_offspring.append(offspring1)
#             genotype_offspring.append(offspring2)

#     return genotype_offspring

# def mutate_genotypes(genotypes: list, mutation_rate=0.1, mutation_strength=0.05):
#     """Mutate the genotypes.
#     Each genotype is made up of 3 arrays of 64 (genotype_size) floats.
#     So we iterate over the genotypes, the arrays, and then mutate the floats.
#     We end up with double the amount of genotypes, the original list and each item mutated."""

#     mutated_genotypes = []
#     for genotype in genotypes:
#         mutated_genotype = []
#         for list1 in genotype:
#             mutated_list1 = []
#             for item in list1:
#                 mutation_size = 0
#                 if RNG.random() <= mutation_rate:
#                     mutation_size = RNG.normal(0, mutation_strength)
#                 mutated_list1.append(item + mutation_size)
#             mutated_genotype.append(mutated_list1)
#         mutated_genotypes.append(mutated_genotype)

#     return mutated_genotypes

def crossover_and_mutation(genotypes: list, scaling_factor=-0.5):
    """FIGURE OUT K-TOURNAMENT SELECTION."""
    """Three parents are selected at random and crossover and mutation creates three children."""
    revde = RevDE(scaling_factor)

    # # for tournament selection
    # n_parents = 3
    # k = (len(genotypes) + n_parents) // 2
    k = 3

    # i stole this from chatGPT so maybe we need to change it up a little but idrk how it works
    indices = RNG.choice(len(genotypes), k, replace=False)
    genotype_parents = [genotypes[i] for i in indices]
    
    # the parents are split into their type, conn, and rot lists and RevDE is
    # applied to each part of the split separately. then they are recombined
    # so the offspring each have a type, conn, and rot list.
    mutated_genotype_offspring = np.zeros((3, 3), dtype=object)
    split_mutated_genotype_offspring = []
    for i in range(3):
        split_genotype_parents = []
        for genotype in genotype_parents:
            split_genotype_parents.append(genotype[i])
        split_mutated_genotype_offspring = revde.mutate(
            np.array(split_genotype_parents[0]),
            np.array(split_genotype_parents[1]),
            np.array(split_genotype_parents[2])
        )
        for j in range(len(split_mutated_genotype_offspring)):
            mutated_genotype_offspring[j,i] = split_mutated_genotype_offspring[j].astype(np.float32)
    mutated_genotype_offspring = mutated_genotype_offspring.tolist()
    
    return mutated_genotype_offspring


def select_survival_genotypes(genotypes: list, gen: int, n: int = POP_SIZE, fitness_function = movement_fitness) -> list:
    fitness_scores = []
    for genotype in genotypes:
        core = construct_core(genotype)
        fitness = calculate_fitness(core, duration=40, fitness_function=fitness_function)
        fitness_scores.append(fitness)
    
    # sort lists from largest fittest value to smallest by adding each value
    # from original ordered list where is belongs in sorted order, for fitness
    # list and parent list at the same time
    fitnesses_sorted, genotypes_sorted = [fitness_scores[0]], [genotypes[0]]
    for i in range(1,len(fitness_scores)):
        added = False

        # if value at index i is larger than value at index j, it is added in
        # front of it in sorted list
        for j in range(len(fitnesses_sorted)):
            if fitness_scores[i] >= fitnesses_sorted[j] and added == False:
                fitnesses_sorted.insert(j, fitness_scores[i])
                genotypes_sorted.insert(j, genotypes[i])
                added = True

        # if value at index i has not been added to sorted list yet, it has
        # smallest fitness value so is added to the back
        if added == False:
            fitnesses_sorted.append(fitness_scores[i])
            genotypes_sorted.append(genotypes[i])
    
    # sorted lists are shortened to return parents and offspring together to
    # original population size
    evolved_genotypes = genotypes_sorted[:n]
    evolved_fitnesses = fitnesses_sorted[:n]

    return evolved_genotypes


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
    timestamp = datetime.datetime.now(tz=datetime.UTC).strftime("%Y%m%d_%H%M%S",)
    save_graph_as_json(
        robot_graph,
        DATA_ROBOTS / f"robot_graph_{timestamp}.json"
    )


def draw_genotype(genotype):
    p_matrices = NDE.forward(genotype)

    # Decode the high-probability graph
    robot_graph: DiGraph[Any] = HPD.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    draw_graph(robot_graph)


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


def initialise_population() -> None:
    genotypes = []
    for _ in range(POP_SIZE):
        genotypes.append(initialise_genotype())
    return genotypes

def main(generations=5):
    # simple initialisation of genotypes and cores
    # genotypes = initialise_population()
    # cores = initialise_cores(genotypes)

    # initialisation with checking for enough movement in beginning
    genotypes = []
    while len(genotypes) < POP_SIZE:
        genotype = initialise_genotype()
        core = construct_core(genotype)
        fitness = calculate_fitness(core,duration=2,fitness_function=fitness_function3)
        print(fitness)
        print(len(genotypes))
        if fitness > 0.1:
            genotypes.append(genotype)

    # print(genotypes[0])

    # trial to see if the different terrains work (to find spawn points)
    # for genotype in [genotypes[0]]:
    #     core = construct_core(genotype)
    #     print(calculate_fitness(core,duration=5,fitness_function=fitness_function2, terrain="flat"))
    #     # print(calculate_fitness(core,duration=5,fitness_function=fitness_function2, terrain="flat"))
    #     # print(calculate_fitness(core,duration=5,fitness_function=fitness_function2, terrain="rough"))
    
    # for genotype in [genotypes[0]]:
    #     core = construct_core(genotype)
    #     print(calculate_fitness(core,duration=5,fitness_function=fitness_function2, terrain="rough"))

    best_genotype = genotypes[0]
    best_fitness = calculate_fitness(construct_core(genotypes[0]), fitness_function=fitness_function3)
    for i in range(generations):
        # apply SNES

        # evolve the robot bodies
        genotypes_and_offspring = genotypes
        for _ in range(5):
            genotypes_and_offspring += crossover_and_mutation(genotypes)
        genotypes = select_survival_genotypes(genotypes_and_offspring, gen=i, fitness_function=fitness_function3)
        fitness = calculate_fitness(construct_core(genotypes[0]), fitness_function=fitness_function3)
        print(fitness)
        if fitness > best_fitness:
            best_genotype = genotypes[0]
            best_fitness = fitness
            print(i)
    best_core = construct_core(best_genotype)
    print(calculate_fitness(best_core, duration=40, fitness_function=fitness_function3, mode="video"))
    draw_genotype(best_genotype)
    save_genotype(best_genotype)


# def test_crossover_and_mutation():
#     hehe = []
#     i = 0
#     for j in range(3):
#         hoho = []
#         for k in range(3):
#             haha = []
#             for l in range(5):
#                 haha.append(i)
#                 i += 1
#             hoho.append(haha)
#         hehe.append(hoho)
#     print(crossover_and_mutation(hehe)[0])


if __name__ == "__main__":
    main(generations=100)
    # test_crossover_and_mutation()