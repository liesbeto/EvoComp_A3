"""
Evolutionary robotics (morphology + controller) using tournament selection + elitism.
Keeps code style close to the original, but adopts your teammate's EA mechanics.
"""

import copy
import random
import numpy as np
import mujoco as mj

from typing import Any, Dict, List

from ariel.simulation.environments import OlympicArena
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder, save_graph_as_json
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.simulation.controllers.controller import Controller
from ariel.utils.tracker import Tracker
from ariel.utils.renderers import video_renderer
from ariel.utils.video_recorder import VideoRecorder

from config import N as NUM_OF_MODULES, pos as SPAWN_POS

import torch

# -------------------- RNG / seeding --------------------
SEED = 33
def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)
seed_all()

RNG = np.random.default_rng(SEED)

# -------------------- EA hyperparams --------------------
POP_SIZE        = 60
GENERATIONS     = 20
GENOTYPE_SIZE   = 64   # size of each NDE input vector
CTRL_GENES_SIZE = 512  # flat controller gene vector
TOURNAMENT_SIZE = 3
ELITE           = 3
MUT_SIG         = 0.08
SIM_DURATION    = 8.0  # seconds

TARGET_POSITION = np.array([5.0, 0.0, 0.5])

# -------------------- Controller gene plumbing --------------------
_current_ctrl_genes: np.ndarray | None = None

def set_current_ctrl_genes(ctrl_genes: np.ndarray) -> None:
    global _current_ctrl_genes
    _current_ctrl_genes = ctrl_genes

def nn_controller(model: mj.MjModel, data: mj.MjData) -> np.ndarray:
    """3-layer MLP; weights come from the flat ctrl gene vector."""
    global _current_ctrl_genes
    input_size  = len(data.qpos)          # no +1 clock in this variant (like teammate)
    hidden_size = 8
    output_size = model.nu

    w1_size = input_size * hidden_size
    w2_size = hidden_size * hidden_size
    w3_size = hidden_size * output_size
    total   = w1_size + w2_size + w3_size

    if _current_ctrl_genes is None:
        # fallback random weights
        w1 = RNG.normal(0, 0.5, (input_size, hidden_size))
        w2 = RNG.normal(0, 0.5, (hidden_size, hidden_size))
        w3 = RNG.normal(0, 0.5, (hidden_size, output_size))
    else:
        genes = _current_ctrl_genes
        if len(genes) < total:
            # pad if fewer genes than needed
            pad = RNG.normal(0, 0.5, total - len(genes))
            genes = np.concatenate([genes, pad])
        idx = 0
        w1 = genes[idx:idx+w1_size].reshape(input_size, hidden_size); idx += w1_size
        w2 = genes[idx:idx+w2_size].reshape(hidden_size, hidden_size); idx += w2_size
        w3 = genes[idx:idx+w3_size].reshape(hidden_size, output_size)

    x  = data.qpos
    h1 = np.tanh(x @ w1)
    h2 = np.tanh(h1 @ w2)
    u  = np.tanh(h2 @ w3)
    return u * np.pi

# -------------------- Fitness --------------------
def fitness_from_history(history_xyz: list[list[float]]) -> float:
    if not history_xyz or len(history_xyz[0]) == 0:
        return float('-inf')
    start_pos = np.array(SPAWN_POS, dtype=float)
    final_pos = np.array(history_xyz[0][-1], dtype=float)
    dist_from_spawn = np.linalg.norm(final_pos - start_pos)
    dist_to_target  = np.linalg.norm(TARGET_POSITION - final_pos)
    return dist_from_spawn - dist_to_target

# -------------------- Individual helpers --------------------
def make_random_individual() -> Dict[str, Any]:
    return {
        "type_p": RNG.random(GENOTYPE_SIZE).astype(np.float32),
        "conn_p": RNG.random(GENOTYPE_SIZE).astype(np.float32),
        "rot_p" : RNG.random(GENOTYPE_SIZE).astype(np.float32),
        "ctrl_genes": RNG.normal(0, 0.3, CTRL_GENES_SIZE).astype(np.float32),
        "fitness": None,
    }

def crossover(a: Dict[str, Any], b: Dict[str, Any]) -> Dict[str, Any]:
    child: Dict[str, Any] = {}
    for key in ("type_p", "conn_p", "rot_p"):
        mask = RNG.random(GENOTYPE_SIZE) < 0.5
        child[key] = np.where(mask, a[key], b[key]).astype(np.float32)
    mask = RNG.random(CTRL_GENES_SIZE) < 0.5
    child["ctrl_genes"] = np.where(mask, a["ctrl_genes"], b["ctrl_genes"]).astype(np.float32)
    child["fitness"] = None
    return child

def mutate(ind: Dict[str, Any]) -> None:
    ind["type_p"] = np.clip(ind["type_p"] + RNG.normal(0, MUT_SIG, ind["type_p"].shape), 0.0, 1.0)
    ind["conn_p"] = np.clip(ind["conn_p"] + RNG.normal(0, MUT_SIG, ind["conn_p"].shape), 0.0, 1.0)
    ind["rot_p"]  = np.clip(ind["rot_p"]  + RNG.normal(0, MUT_SIG, ind["rot_p"].shape), 0.0, 1.0)
    ind["ctrl_genes"] = ind["ctrl_genes"] + RNG.normal(0, MUT_SIG, ind["ctrl_genes"].shape)

def tournament_select(pop: List[Dict[str, Any]]) -> Dict[str, Any]:
    k = min(TOURNAMENT_SIZE, len(pop))
    aspirants = RNG.choice(pop, size=k, replace=False)
    aspirants = sorted(aspirants, key=lambda ind: ind["fitness"] if ind["fitness"] is not None else -1e9, reverse=True)
    return copy.deepcopy(aspirants[0])

# -------------------- Evaluation --------------------
def evaluate(ind: Dict[str, Any], sim_duration: float = SIM_DURATION) -> float:
    try:
        # decode morphology
        nde = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
        hpd = HighProbabilityDecoder(NUM_OF_MODULES)

        p_mats = nde.forward([ind["type_p"], ind["conn_p"], ind["rot_p"]])
        graph  = hpd.probability_matrices_to_graph(p_mats[0], p_mats[1], p_mats[2])

        # build mujoco
        robot_spec = construct_mjspec_from_graph(graph)
        world = OlympicArena()
        world.spawn(robot_spec.spec, spawn_position=SPAWN_POS)
        model = world.spec.compile()
        data  = mj.MjData(model)

        if model.nu == 0:
            ind["fitness"] = float('-inf')
            return ind["fitness"]

        # tracker + controller
        tracker = Tracker(mujoco_obj_to_find=mj.mjtObj.mjOBJ_GEOM, name_to_bind="core")
        tracker.setup(world.spec, data)

        set_current_ctrl_genes(ind["ctrl_genes"])
        ctrl = Controller(controller_callback_function=nn_controller, tracker=tracker)

        mj.set_mjcb_control(None)
        mj.mj_resetData(model, data)
        mj.set_mjcb_control(lambda m, d: ctrl.set_control(m, d))

        # run headless
        t_end = data.time + sim_duration
        while data.time < t_end:
            mj.mj_step(model, data)

        # fitness
        if "xpos" not in tracker.history:
            ind["fitness"] = float('-inf')
        else:
            ind["fitness"] = fitness_from_history(tracker.history["xpos"])
        return ind["fitness"]
    except Exception:
        ind["fitness"] = float('-inf')
        return ind["fitness"]

# -------------------- EA loop --------------------
def run_ea() -> Dict[str, Any]:
    seed_all()
    pop = [make_random_individual() for _ in range(POP_SIZE)]

    # initial evaluation
    for ind in pop:
        evaluate(ind)

    for gen in range(1, GENERATIONS + 1):
        pop.sort(key=lambda x: x["fitness"], reverse=True)
        best = pop[0]["fitness"]
        valid = [i["fitness"] for i in pop if i["fitness"] != float('-inf')]
        avg = float(np.mean(valid)) if valid else float('-inf')
        print(f"Gen {gen:02d} | Best {best:.3f} | Avg {avg:.3f} | Valid {len(valid)}/{POP_SIZE}")

        new_pop = pop[:ELITE]  # elitism

        while len(new_pop) < POP_SIZE:
            pa = tournament_select(pop)
            pb = tournament_select(pop)
            child = crossover(pa, pb)
            mutate(child)
            evaluate(child)
            new_pop.append(child)

        pop = new_pop

    best_individual = max(pop, key=lambda x: x["fitness"])
    print(f"Final best fitness: {best_individual['fitness']:.3f}")
    return best_individual

if __name__ == "__main__":
    best = run_ea()
