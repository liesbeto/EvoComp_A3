import datetime

from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import mujoco as mj

if TYPE_CHECKING:
    from networkx import DiGraph

from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import (
    HighProbabilityDecoder,
    save_graph_as_json)

# --- RANDOM GENERATOR SETUP --- #
SEED = 42
RNG = np.random.default_rng(SEED)

NUM_OF_MODULES = 30
NDE = NeuralDevelopmentalEncoding(number_of_modules=NUM_OF_MODULES)
HPD = HighProbabilityDecoder(NUM_OF_MODULES)


def construct_core(genotype):

    mj.set_mjcb_control(None)

    p_matrices = NDE.forward(genotype)
    robot_graph: DiGraph[Any] = HPD.probability_matrices_to_graph(
        p_matrices[0],
        p_matrices[1],
        p_matrices[2],
    )

    core = construct_mjspec_from_graph(robot_graph)
    return mj.MjSpec.from_string(core.spec.to_xml())


def save_genotype(genotype, folder):
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
    filename = f"graph_{timestamp}.json"
    save_graph_as_json(
        robot_graph,
        folder / filename
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
