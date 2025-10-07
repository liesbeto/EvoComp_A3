"""
Assignment 3: Evolutionary Robotics with Neural Developmental Encoding

This script implements a complete evolutionary robotics system that co-evolves:
1. Robot morphology using Neural Developmental Encoding (NDE)
2. Neural network controllers for locomotion

Key Components:
- NDE-based morphology evolution: Evolves 3 input vectors that generate probability matrices
- Neural network controller evolution: Evolves weights for a 3-layer feedforward network
- Enhanced fitness function: Rewards movement toward target while penalizing immobility
- Tournament selection with elitism for robust evolution

Usage:
    python A3_template.py

The system will:
1. Initialize a population of random robot morphologies and controllers
2. Evaluate each robot's fitness in a physics simulation
3. Evolve the population over multiple generations
4. Visualize the best robot and save its configuration

Author: [Your Team]
Date: October 2025
"""

import datetime
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal, Dict, List
import copy

import matplotlib.pyplot as plt
import mujoco as mj
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
)
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.simulation.controllers.controller import Controller
from ariel.simulation.environments import OlympicArena
from ariel.utils.renderers import single_frame_renderer, video_renderer
from ariel.utils.runners import simple_runner
from ariel.utils.tracker import Tracker
from ariel.utils.video_recorder import VideoRecorder

# Type Checking
if TYPE_CHECKING:
    from networkx import DiGraph

# Type Aliases
ViewerTypes = Literal["launcher", "video", "simple", "no_control", "frame"]

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
NUM_OF_MODULES = 20              # Standard module count
TARGET_POSITION = [5, 0, 0.5]

# --- EVOLUTIONARY ALGORITHM HYPERPARAMETERS --- #
POP_SIZE = 100                   # Population size (balanced for performance vs time)
GENERATIONS = 30                 # Number of generations (sufficient for convergence)
GENOTYPE_SIZE = 64               # NDE input vector size (required by neural network)
CTRL_GENES_SIZE = 512            # Neural network controller genes (3-layer network)
TOURNAMENT_SIZE = 3              # Tournament selection size
ELITE = 5                        # Elite individuals preserved per generation (5% elitism)
MUT_SIG = 0.08                   # Gaussian mutation standard deviation
SIM_DURATION = 8                 # Simulation time per robot evaluation (seconds)

def fitness_function(history: list[float]) -> float:
    """
    Enhanced fitness function that rewards robots for moving toward the target.
    
    Args:
        history: List of robot positions [x, y, z] over time
        
    Returns:
        float: Fitness score (higher is better)
        
    Fitness combines:
        - Distance traveled from spawn point (reward exploration)  
        - Negative distance to target (reward goal-seeking)
        - This encourages robots that move toward the target
    """
    # Get positions
    start_pos = np.array(SPAWN_POS)      # [-0.8, 0, 0.1]
    target_pos = np.array(TARGET_POSITION)  # [5, 0, 0.5]
    final_pos = np.array(history[-1])    # Robot's final position
    
    # Simple approach: distance from spawn minus distance to target
    distance_from_spawn = np.linalg.norm(final_pos - start_pos)
    distance_to_target = np.linalg.norm(target_pos - final_pos)
    
    # Combined fitness
    fitness = distance_from_spawn - distance_to_target
    
    return fitness


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
    plt.show()


# Global variable to store current individual's control genes
_current_ctrl_genes = None

def set_current_ctrl_genes(ctrl_genes: np.ndarray) -> None:
    """Set the control genes for the current individual being evaluated."""
    global _current_ctrl_genes
    _current_ctrl_genes = ctrl_genes

def nn_controller(
    model: mj.MjModel,
    data: mj.MjData,
) -> npt.NDArray[np.float64]:
    """Neural network controller using evolved weights from individual's ctrl genes."""
    global _current_ctrl_genes
    
    # Simple 3-layer neural network
    input_size = len(data.qpos)
    hidden_size = 8
    output_size = model.nu
    
    # Calculate weight matrix sizes
    w1_size = input_size * hidden_size
    w2_size = hidden_size * hidden_size  
    w3_size = hidden_size * output_size
    total_weights = w1_size + w2_size + w3_size
    
    if _current_ctrl_genes is None:
        console.log("WARNING: No control genes set, using random weights!")
        w1 = RNG.normal(loc=0.0138, scale=0.5, size=(input_size, hidden_size))
        w2 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, hidden_size))
        w3 = RNG.normal(loc=0.0138, scale=0.5, size=(hidden_size, output_size))
    else:
        # Use evolved control genes as neural network weights
        if len(_current_ctrl_genes) < total_weights:
            console.log(f"WARNING: Not enough control genes ({len(_current_ctrl_genes)}) for network weights ({total_weights})")
            # Pad with random weights if not enough genes
            padded_genes = np.concatenate([
                _current_ctrl_genes,
                RNG.normal(0, 0.5, total_weights - len(_current_ctrl_genes))
            ])
        else:
            padded_genes = _current_ctrl_genes
            
        # Extract weight matrices from evolved genes
        idx = 0
        w1 = padded_genes[idx:idx+w1_size].reshape(input_size, hidden_size)
        idx += w1_size
        w2 = padded_genes[idx:idx+w2_size].reshape(hidden_size, hidden_size)
        idx += w2_size
        w3 = padded_genes[idx:idx+w3_size].reshape(hidden_size, output_size)

    # Get inputs - positions of actuator motors (hinges)
    inputs = data.qpos

    # Forward pass through the network layers
    layer1 = np.tanh(np.dot(inputs, w1))
    layer2 = np.tanh(np.dot(layer1, w2))
    outputs = np.tanh(np.dot(layer2, w3))

    # Scale the outputs to appropriate range
    return outputs * np.pi


def experiment(
    robot: Any,
    controller: Controller,
    duration: int = 15,
    mode: ViewerTypes = "viewer",
) -> None:
    """Run the simulation with random movements."""
    # ==================================================================== #
    # Initialise controller to controller to None, always in the beginning.
    mj.set_mjcb_control(None)  # DO NOT REMOVE

    # Initialise world
    world = OlympicArena()

    # Spawn robot in the world
    world.spawn(robot.spec, spawn_position=SPAWN_POS)

    # Generate the model and data
    model = world.spec.compile()
    data = mj.MjData(model)

    # Reset state and time of simulation
    mj.mj_resetData(model, data)

    # Pass the model and data to the tracker
    if controller.tracker is not None:
        controller.tracker.setup(world.spec, data)  # Ensure tracker is set up correctly

    # Set the control callback function
    args: list[Any] = []  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!
    kwargs: dict[Any, Any] = {}  # IF YOU NEED MORE ARGUMENTS ADD THEM HERE!

    mj.set_mjcb_control(
        lambda m, d: controller.set_control(m, d, *args, **kwargs),
    )

    # ------------------------------------------------------------------ #
    match mode:
        case "simple":
            simple_runner(model, data, duration=duration)
        case "frame":
            save_path = str(DATA / "robot.png")
            single_frame_renderer(model, data, save=True, save_path=save_path)
        case "video":
            path_to_video_folder = str(DATA / "videos")
            video_recorder = VideoRecorder(output_folder=path_to_video_folder)
            video_renderer(model, data, duration=duration, video_recorder=video_recorder)
        case "launcher":
            viewer.launch(model=model, data=data)
        case "no_control":
            mj.set_mjcb_control(None)
            viewer.launch(model=model, data=data)
    # ==================================================================== #


def test_robot_movement_potential(ind: Dict[str, Any]) -> bool:
    """Enhanced test for robot movement potential with multiple validation checks."""
    try:
        # Build the robot
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        p_matrices = nde.forward([ind["type_p"], ind["conn_p"], ind["rot_p"]])
        
        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        graph = hpd.probability_matrices_to_graph(p_matrices[0], p_matrices[1], p_matrices[2])
        
        robot_spec = construct_mjspec_from_graph(graph)
        if not hasattr(robot_spec.spec, 'compile'):
            return False  # Can't build robot
            
        # Test 1: Check if robot has at least one actuated joint
        # Count HINGE modules in the robot structure
        hinge_count = sum(1 for node in graph.nodes() if graph.nodes[node].get('type') == 'HINGE')
        if hinge_count < 1:  # Need at least 1 hinge for any movement
            return False
            
        # Test 2: Quick movement test with multiple controllers
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        
        world = OlympicArena()
        world.spawn(robot_spec.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()
        data = mj.MjData(model)
        
        # Test with 3 different control patterns
        movement_tests = []
        
        for test_id in range(3):
            mj.mj_resetData(model, data)  # Reset for each test
            tracker.setup(world.spec, data)
            
            # Different control patterns
            if test_id == 0:
                # Sinusoidal movement
                def test_controller(model, data):
                    t = data.time
                    return np.sin(t * 2 * np.pi) * np.ones(model.nu) * 0.5
            elif test_id == 1:
                # Random movement
                def test_controller(model, data):
                    return RNG.uniform(-0.5, 0.5, model.nu)
            else:
                # Constant movement
                def test_controller(model, data):
                    return np.ones(model.nu) * 0.3
            
            ctrl = Controller(controller_callback_function=test_controller, tracker=tracker)
            mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))
            
            # Short simulation (1.5 seconds)
            simple_runner(model, data, duration=1.5)
            
            if 'xpos' in tracker.history and len(tracker.history["xpos"][0]) > 1:
                start_pos = np.array(tracker.history["xpos"][0][0])
                end_pos = np.array(tracker.history["xpos"][0][-1])
                distance_moved = np.linalg.norm(end_pos - start_pos)
                movement_tests.append(distance_moved)
            else:
                movement_tests.append(0.0)
        
        # Robot passes if it shows movement in at least 1 out of 3 tests
        # (More lenient to allow more robots through)
        successful_tests = sum(1 for dist in movement_tests if dist > 0.02)
        avg_movement = np.mean(movement_tests)
        
        return successful_tests >= 1 and avg_movement > 0.01
        
    except Exception as e:
        # Log the specific error for debugging
        console.log(f"Movement test failed: {e}")
        return False  # Any error means non-learner


def make_random_individual() -> Dict[str, Any]:
    """
    Generate a random individual for the evolutionary algorithm.
    
    Creates random vectors for NDE morphology generation and neural network genes.
    According to assignment requirements, we evolve the INPUT VECTORS to NDE,
    not the output matrices.
    
    Returns:
        Dict containing:
            - type_p: Vector for module type probabilities (size 64)
            - conn_p: Vector for connection probabilities (size 64)  
            - rot_p: Vector for rotation probabilities (size 64)
            - ctrl_genes: Neural network controller genes (size 512)
            - fitness: None (to be evaluated)
    """
    # Create random developmental parameter VECTORS (not matrices!)
    # NDE takes vectors as input and converts them to matrices internally
    type_p = RNG.random(GENOTYPE_SIZE)      # Vector for type probabilities
    conn_p = RNG.random(GENOTYPE_SIZE)      # Vector for connection probabilities  
    rot_p = RNG.random(GENOTYPE_SIZE)       # Vector for rotation probabilities
    
    # Create random controller genes
    ctrl_genes = RNG.normal(0, 0.3, CTRL_GENES_SIZE)
    
    # Create candidate individual
    candidate = {
        "type_p": type_p,
        "conn_p": conn_p,
        "rot_p": rot_p,
        "ctrl_genes": ctrl_genes,
        "fitness": None
    }
    
    return candidate

def evaluate(ind: Dict[str, Any], sim_duration: int = SIM_DURATION) -> float:
    """
    Evaluate an individual by running physics simulation.
    
    Process:
    1. Use NDE to convert input vectors to probability matrices
    2. Use HighProbabilityDecoder to generate robot morphology
    3. Create neural network controller from evolved genes
    4. Simulate robot in MuJoCo physics environment
    5. Calculate fitness based on movement toward target
    
    Args:
        ind: Individual dictionary with morphology vectors and controller genes
        sim_duration: Simulation time in seconds
        
    Returns:
        float: Fitness score (higher is better, -inf if robot compilation fails)
        
    Note: Some randomly generated morphologies may fail MuJoCo compilation.
    This is expected behavior in evolutionary robotics.
    """
    try:
        # Morphology decode using NDE
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        p_matrices = nde.forward([ind["type_p"], ind["conn_p"], ind["rot_p"]])

        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        graph = hpd.probability_matrices_to_graph(p_matrices[0], p_matrices[1], p_matrices[2])

        # Construct Mujoco model from graph
        robot_spec = construct_mjspec_from_graph(graph)  # Create CoreModule from graph

        # Check the type of robot_spec.spec (the actual MuJoCo specification)
        if not hasattr(robot_spec.spec, 'compile'):
            console.log(f"Error: robot_spec.spec does not have 'compile' method. Type: {type(robot_spec.spec)}")
            ind["fitness"] = float('-inf')  # Set fitness to a low value on error
            return ind["fitness"]

        # Initialize the tracker with necessary parameters (track the core geometry)
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")

        # Set the control genes for the neural network controller
        set_current_ctrl_genes(ind["ctrl_genes"])
        
        # Create the controller
        ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)

        # Run the experiment using simple_runner
        world = OlympicArena()
        world.spawn(robot_spec.spec, spawn_position=SPAWN_POS)

        # Generate the model from the world spec (same pattern as in experiment function)
        model = world.spec.compile()

        data = mj.MjData(model)

        # Reset state and time of simulation
        mj.mj_resetData(model, data)

        # Set up the tracker with the world spec and data
        tracker.setup(world.spec, data)

        # Set the control callback function
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        # Use simple_runner to run the simulation (headless simulation as specified)
        simple_runner(model, data, duration=sim_duration)

        # Check if 'xpos' is in tracker.history
        if 'xpos' not in tracker.history:
            console.log("Error: 'xpos' not found in tracker history.")
            console.log(f"Tracker history contents: {tracker.history}")  # Log the contents of the tracker history
            ind["fitness"] = float('-inf')  # Set fitness to a low value on error
            return ind["fitness"]

        # Calculate fitness
        fitness = fitness_function(tracker.history["xpos"][0])
        ind["fitness"] = fitness
        return fitness

    except Exception as e:
        import traceback
        console.log(f"Error during evaluation: {e}")
        console.log(f"Full traceback: {traceback.format_exc()}")
        ind["fitness"] = float('-inf')  # Set fitness to a low value on error
        return ind["fitness"]

def tournament_select(pop: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Return a selected individual (copy) by tournament."""
    aspirants = RNG.choice(pop, size=min(TOURNAMENT_SIZE, len(pop)), replace=False)
    aspirants = sorted(aspirants, key=lambda ind: ind["fitness"] if ind["fitness"] is not None else -1e9, reverse=True)
    return copy.deepcopy(aspirants[0])

def crossover(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    """
    Uniform crossover between two parent individuals.
    
    For each gene position, randomly selects from parent A or B with 50% probability.
    Operates on both morphology vectors (NDE inputs) and controller genes.
    
    Args:
        a, b: Parent individuals
        
    Returns:
        Dict: Child individual with mixed genes from both parents
    """
    child = {}
    for key in ("type_p", "conn_p", "rot_p"):
        mask = RNG.random(GENOTYPE_SIZE) < 0.5
        child[key] = np.where(mask, a[key], b[key]).astype(np.float32)
    mask = RNG.random(CTRL_GENES_SIZE) < 0.5
    child["ctrl_genes"] = np.where(mask, a["ctrl_genes"], b["ctrl_genes"]).astype(np.float32)
    child["fitness"] = None
    return child

def mutate(ind: Dict[str, Any]) -> None:
    """
    Apply Gaussian mutation to an individual (in-place).
    
    Adds Gaussian noise to all genes:
    - Morphology vectors: Clipped to [0,1] range (valid probabilities)
    - Controller genes: Unbounded (neural network weights can be any value)
    
    Args:
        ind: Individual to mutate (modified in place)
    """
    ind["type_p"] = np.clip(ind["type_p"] + RNG.normal(0, MUT_SIG, ind["type_p"].shape), 0.0, 1.0)
    ind["conn_p"] = np.clip(ind["conn_p"] + RNG.normal(0, MUT_SIG, ind["conn_p"].shape), 0.0, 1.0)
    ind["rot_p"] = np.clip(ind["rot_p"] + RNG.normal(0, MUT_SIG, ind["rot_p"].shape), 0.0, 1.0)
    ind["ctrl_genes"] += RNG.normal(0, MUT_SIG, ind["ctrl_genes"].shape)

def run_ea():
    """
    Run the evolutionary algorithm for robot optimization.
    
    Algorithm:
    1. Initialize random population of robot morphologies and controllers
    2. Evaluate each individual's fitness through physics simulation
    3. For each generation:
       - Select parents using tournament selection
       - Create offspring through crossover and mutation
       - Preserve elite individuals
       - Replace population with new generation
    4. Return best individual found
    
    Uses co-evolution of:
    - Morphology: 3 NDE input vectors (type, connection, rotation probabilities)
    - Control: Neural network weights for locomotion controller
    
    Returns:
        Dict: Best individual found during evolution
    """
    import time
    start_time = time.time()
    
    console.log(f"🚀 Starting evolution with POP_SIZE={POP_SIZE}, GENERATIONS={GENERATIONS}, SIM_DURATION={SIM_DURATION}s")
    
    # Initialize population
    pop = [make_random_individual() for _ in range(POP_SIZE)]
    
    # Initial evaluation
    console.log("Evaluating initial population...")
    for ind in pop:
        evaluate(ind)

    for gen in range(1, GENERATIONS + 1):
        pop.sort(key=lambda x: x["fitness"], reverse=True)
        best_fitness = pop[0]["fitness"]
        
        # Enhanced statistics with success rate
        valid_fitnesses = [ind["fitness"] for ind in pop if ind["fitness"] != float('-inf')]
        avg_fitness = np.mean(valid_fitnesses) if valid_fitnesses else float('-inf')
        valid_count = len(valid_fitnesses)
        success_rate = (valid_count / POP_SIZE) * 100
        console.log(f"Generation {gen}: Best = {best_fitness:.3f}, Avg = {avg_fitness:.3f}, Valid robots = {valid_count}/{POP_SIZE} ({success_rate:.1f}%)")

        new_pop = pop[:ELITE]  # Elitism

        while len(new_pop) < POP_SIZE:
            parent_a = tournament_select(pop)
            parent_b = tournament_select(pop)
            child = crossover(parent_a, parent_b)
            mutate(child)
            evaluate(child)
            new_pop.append(child)

        pop = new_pop

    # Final best
    best_individual = max(pop, key=lambda x: x["fitness"])
    
    # Calculate and display timing information
    end_time = time.time()
    total_time = end_time - start_time
    console.log(f"Evolution completed in {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
    console.log(f"Final best fitness: {best_individual['fitness']}")

    # Save and visualize the best robot
    save_and_visualize_best_robot(best_individual)
    
    return best_individual

def save_and_visualize_best_robot(best_individual: Dict[str, Any]) -> None:
    """Save the best robot as JSON and visualize it."""
    try:
        # Reconstruct the robot from the best individual's genotype
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        p_matrices = nde.forward([best_individual["type_p"], best_individual["conn_p"], best_individual["rot_p"]])

        hpd = HighProbabilityDecoder(NUM_OF_MODULES)
        graph = hpd.probability_matrices_to_graph(p_matrices[0], p_matrices[1], p_matrices[2])

        # Save the graph as JSON
        json_filename = DATA_ROBOTS / "final_best.json"
        save_graph_as_json(graph, json_filename)
        console.log(f"Saved best robot graph to: {json_filename}")

        # Construct the robot for visualization
        robot_spec = construct_mjspec_from_graph(graph)

        # Set the control genes for the neural network controller
        set_current_ctrl_genes(best_individual["ctrl_genes"])
        
        # Create controller with tracker for visualization
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)

        # Visualize the robot using the launcher
        console.log("Launching MuJoCo viewer to visualize the best robot...")
        experiment(robot=robot_spec, controller=ctrl, duration=15, mode="launcher")

        # After closing the launcher, show the robot's path
        if tracker.history and "xpos" in tracker.history:
            console.log("Showing robot path visualization...")
            show_xpos_history(tracker.history["xpos"][0])
        else:
            console.log("No tracking data available to show path")

    except Exception as e:
        console.log(f"Error saving/visualizing best robot: {e}")

def main() -> None:
    """Entry point - runs evolutionary robotics optimization."""
    run_ea()

if __name__ == "__main__":
    main()
