import datetime
import csv
from pathlib import Path
import os

POP_SIZE = 8

from initialise import initialise_genotype, construct_core, save_genotype
from body_utils import calculate_fitness, crossover_and_mutation, pick_best_robots
from brain_utils import evolution_brain, experiment_brain
from plot import make_plot_two_y
from fitness_functions import fitness_function_simple, fitness_function_complex


CWD = Path.cwd()

EVOLUTION_NAME = "example"
EVOLUTION_FOLDER = CWD / EVOLUTION_NAME
EVOLUTION_FOLDER.mkdir(exist_ok=True)
EVOLUTION_GRAPHS_FOLDER = EVOLUTION_FOLDER / "graphs"
EVOLUTION_GRAPHS_FOLDER.mkdir(exist_ok=True)

BASELINE_NAME = f"{EVOLUTION_NAME}_baseline"
BASELINE_FOLDER = CWD / BASELINE_NAME
BASELINE_FOLDER.mkdir(exist_ok=True)
BASELINE_GRAPHS_FOLDER = BASELINE_FOLDER / "graphs"
BASELINE_GRAPHS_FOLDER.mkdir(exist_ok=True)


def evolutionary_algorithm(body_gens, min_brain_gens=1, max_brain_gens=1):
    # order in dictionary: place in population : [genotype, graph_filename, core, best_policy, best_policy_filename, fitness]
    robots = {}

    # initialise robots with enough general movement to population
    while len(robots) < POP_SIZE:
        genotype = initialise_genotype()
        core = construct_core(genotype)
        fitness = calculate_fitness(core, duration=4, fitness_function=fitness_function_simple)
        if fitness > 0.1:
            # store actual DATA_ROBOTS path for the saved genotype
            robots[len(robots)] = [genotype, f"{EVOLUTION_GRAPHS_FOLDER}/{save_genotype(genotype, EVOLUTION_GRAPHS_FOLDER)}", core]

    # prepare incremental CSV writer so results are available even on interrupt
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = EVOLUTION_FOLDER / f"max_fitnesses_{EVOLUTION_NAME}_{timestamp}.csv"
    csvfile = open(save_path, "w", newline="")
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["generation", "max_fitness"])

    for gen in range(body_gens):
        # create 3 offspring
        robots = crossover_and_mutation(robots, POP_SIZE, EVOLUTION_GRAPHS_FOLDER)

        # Linear ramp up of brain generations
        brain_gens = min_brain_gens + (max_brain_gens - min_brain_gens) * (gen / body_gens)
        brain_gens = int(brain_gens)  # Convert to integer

        # apply XNES to population and calculate fitness
        for i in range(len(robots)):
            if len(robots[i]) < 4:
                label = os.path.splitext(os.path.basename(robots[i][1]))[0]
                best_policy, best_policy_filename = evolution_brain(f"{label}_brain", robots[i][1], folder=EVOLUTION_NAME, generations=brain_gens)
                robots[i].append(best_policy)
                robots[i].append(best_policy_filename)
                fitness = experiment_brain(robots[i][3], robots[i][1])
                robots[i].append(fitness)

        # pick 5 best robots
        robots = pick_best_robots(robots, POP_SIZE)

        # write incremental result immediately
        csv_writer.writerow([gen, robots[0][-1]])
        csvfile.flush()

    csvfile.close()

    return save_path


def baseline(body_gens):
    # order in dictionary: place in population : [genotype, graph_filename, core, best_policy, best_policy_filename, fitness]
    robots = {}

    # initialise robots with enough general movement to population
    while len(robots) < POP_SIZE:
        genotype = initialise_genotype()
        core = construct_core(genotype)
        flat_fitness = calculate_fitness(core, fitness_function_complex, duration=10, terrain="flat")
        core = construct_core(genotype)
        rugged_fitness = calculate_fitness(core, fitness_function_complex, duration=10, terrain="rugged")
        robots[len(robots)] = [genotype, f"{BASELINE_GRAPHS_FOLDER}/{save_genotype(genotype, BASELINE_GRAPHS_FOLDER)}", core, flat_fitness + rugged_fitness]

    # prepare incremental CSV writer so results are available even on interrupt
    timestamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    save_path = BASELINE_FOLDER / f"max_fitnesses_{BASELINE_NAME}_{timestamp}.csv"
    csvfile = open(save_path, "w", newline="")
    csv_writer = csv.writer(csvfile)
    csv_writer.writerow(["generation", "max_fitness"])

    for gen in range(body_gens):
        # create 3 random offspring
        for _ in range(3):
            genotype = initialise_genotype()
            core = construct_core(genotype)
            flat_fitness = calculate_fitness(core, fitness_function_complex, duration=10, terrain="flat")
            core = construct_core(genotype)
            rugged_fitness = calculate_fitness(core, fitness_function_complex, duration=10, terrain="rugged")
            robots[len(robots)] = [genotype, f"{BASELINE_GRAPHS_FOLDER}/{save_genotype(genotype, BASELINE_GRAPHS_FOLDER)}", core, flat_fitness + rugged_fitness]

        # pick 5 best robots
        robots = pick_best_robots(robots, POP_SIZE)

        # write incremental result immediately
        csv_writer.writerow([gen, robots[0][-1]])
        csvfile.flush()

    csvfile.close()
    
    return save_path


if __name__ == "__main__":
    body_gens = 2
    evolution_path = evolutionary_algorithm(body_gens=body_gens, min_brain_gens=1, max_brain_gens=1)
    baseline_path = baseline(body_gens=body_gens)
    make_plot_two_y(evolution_path, baseline_path, "max_fitness", save_path=f"{EVOLUTION_FOLDER}/{BASELINE_NAME}_plot.pdf")
