"""
Simple brain evolution for Assignment 3.
Evolves only the neural controller for one fixed body.
"""

import numpy as np
import mujoco as mj
from individual import run_individual
from ariel.simulation.environments import OlympicArena
from ariel.ec.genotypes.nde import NeuralDevelopmentalEncoding
from ariel.body_phenotypes.robogen_lite.decoders.hi_prob_decoding import HighProbabilityDecoder
from ariel.body_phenotypes.robogen_lite.constructor import construct_mjspec_from_graph
from config import N, pos

import random, numpy as np
import networkx as nx
import torch  

SEED = 33
# Uses the same seed for all the stochastity 
def seed_all(s=SEED):
    random.seed(s)
    np.random.seed(s)
    torch.manual_seed(s)

seed_all()
nde = NeuralDevelopmentalEncoding(number_of_modules=N)
hpd = HighProbabilityDecoder(N)


def flatten_weights(weights):
    """ help function to make the lists of the weights into one 1d list"""
    return np.concatenate([w.flatten() for w in weights])

def unflatten_weights(genome, input_size, hidden_size, output_size):
    """Convers the long 1d list back into seperate lists"""
    i_end = input_size * hidden_size
    h_end = i_end + hidden_size * hidden_size
    o_end = h_end + hidden_size * output_size
    w1 = genome[:i_end].reshape(input_size, hidden_size)
    w2 = genome[i_end:h_end].reshape(hidden_size, hidden_size)
    w3 = genome[h_end:o_end].reshape(hidden_size, output_size)
    return (w1, w2, w3)


def make_body():
    """Build one random fixed body"""
    genotype_size = 64
    type_genes = np.random.rand(genotype_size).astype(np.float32)
    conn_genes = np.random.rand(genotype_size).astype(np.float32)
    rot_genes = np.random.rand(genotype_size).astype(np.float32)
    return [type_genes, conn_genes, rot_genes]

    
def decode_robot_spec(body_genes):
    """ Uses the 3 lists of length 64 body_genes and NDE to return robot """
    # given body_genes NDE predicts probability tensors for N module robot
    #nde = NeuralDevelopmentalEncoding(number_of_modules=N) 
    # Run gnome through NDE output is probability matrix 
    p_matrices = nde.forward(body_genes)
    # Makes the probabilities into robotgraph 
    #hpd = HighProbabilityDecoder(N)
    robot_graph = hpd.probability_matrices_to_graph(*p_matrices) # use * to pass elements of list one by one
    # turn graph into mujoco specification
    robot = construct_mjspec_from_graph(robot_graph)
    spec = robot.spec
    model = spec.compile()  # compiles into a MuJoCo model
    return model


def dims_from_spec(model, hidden_size):
    """ Gets dimensions for neural network based on the body strucutre of robot"""
    # create mujoco data structure from model
    data  = mj.MjData(model)

    # input amount of joints plus sinus clock output amount of actuators 
    input_size  = int(data.qpos.size + 1)
    output_size = int(model.nu)

    # how many total numbers our neural network would have
    genome_size = (input_size * hidden_size+
        hidden_size * hidden_size +
        hidden_size * output_size
    )
    return input_size, output_size, genome_size

def evolve_brain(body_genes,generations, pop_size, hidden_size=8, sigma=0.1):

    robot_spec_once = decode_robot_spec(body_genes)
    input_size, output_size, genome_size= dims_from_spec(robot_spec_once, hidden_size)

    # create initial population
    population = [np.random.randn(genome_size) for _ in range(pop_size)]

    # loop over generations and then over the populations
    for gen in range(generations):
        fitnesses = []
        for genome in population:
            weights = unflatten_weights(genome, input_size, hidden_size, output_size)
            fit = run_individual(body_genes, weights, view=False)
            fitnesses.append(fit)

        # get best fitness via index
        best_idx = np.argmax(fitnesses)
        best_fit = fitnesses[best_idx]
        print(f"Gen {gen}: best fitness = {best_fit:.3f}")
        
        best_genome = population[best_idx]

        # creates next generation with mutation used for now simple ES
        population = [best_genome + sigma * np.random.randn(genome_size) for _ in range(pop_size)]

    return best_genome, body_genes, (input_size, hidden_size, output_size)

def evolve_bodies(num_generations=5, body_pop_size=5, brain_gens=5, brain_pop=6):
    """ Evolves bodies and for each body evolves the brain as well"""
    # Create random boddies
    bodies = [make_body() for _ in range(body_pop_size)]
    hidden_size = 8

    for gen in range(num_generations):
        body_fitnesses = []

        for i, body in enumerate(bodies):
            robot_spec = decode_robot_spec(body)  
            input_size, output_size, _= dims_from_spec(robot_spec,hidden_size) 

            # runs brain evolution for specific body 
            best_brain, _, _ = evolve_brain(body,brain_gens,brain_pop,hidden_size)
            
            best_weights = unflatten_weights(best_brain, input_size, hidden_size, output_size)
            fit = run_individual(body, best_weights)
            body_fitnesses.append(fit)

        # Rank and select the best bodies
        best_idx = np.argsort(body_fitnesses)[::-1]
        best_bodies = [bodies[i] for i in best_idx[:2]]  # keep top 2
        print(f"  Best body fitness: {body_fitnesses[best_idx[0]]:.3f}")

        # Mutate others 
        new_bodies = []
        for b in best_bodies:
            new_bodies.append(b)
            for _ in range(2):
                mutated = [np.clip(g + 0.1 * np.random.randn(*g.shape), 0, 1) for g in b]
                new_bodies.append(mutated)

        # Add 1 random fresh body each gen
        new_bodies.append(make_body())
        bodies = new_bodies[:body_pop_size]

if __name__ == "__main__":
    seed_all()
    evolve_bodies(num_generations=5, body_pop_size=5, brain_gens=5, brain_pop=6)