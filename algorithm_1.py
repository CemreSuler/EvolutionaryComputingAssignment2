# NOTE importing Numba to make sure that important calculations happen much faster
from numba import njit

from evoman.environment import Environment

# NOTE renamed to adhere to PIP conventions
from demo_controller import player_controller as PlayerController

import numpy as np

import time
import os

## SETTINGS
# PyGame Settings
HEADLESS = True
EXPERIMENT_NAME = "Algorithm1"

# Model settings
NUM_HIDDEN_NEURONS = 10
POPULATION_SIZE = 1000
TRAINING_GENERATIONS = 50
MUTATION_RATE = 0.2  # NOTE We want to somehow modify this rate I believe
## END OF SETTINGS


if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# NOTE contact_hurt already set as default
ENVIRONMENT = Environment(
    experiment_name=EXPERIMENT_NAME,
    enemies=[1, 2, 3, 4, 5, 6, 7, 8],
    multiplemode="yes",
    playermode="ai",
    player_controller=PlayerController(NUM_HIDDEN_NEURONS),
    enemymode="static",
    level=2,
    speed="fastest",
    visuals=False,
    logs="off",
)

# NOTE ensures that all values are returned, to ensure all values are stored
ENVIRONMENT.cons_multi = lambda values: values
NUM_VARS = (ENVIRONMENT.get_num_sensors() + 1) * NUM_HIDDEN_NEURONS + (
    NUM_HIDDEN_NEURONS + 1
) * 5


# FIXME need to create a function that combines the different values in some sort of good way, for now just taking the mean
# NOTE numba optimization not possible
def single_simulation(x_1):
    fitness, player_life, enemy_life, time = ENVIRONMENT.play(pcont=x_1)
    return float(np.mean(fitness))


# NOTE numba optimization not possible
def run_all_simulations(x):
    return np.array(list(map(lambda x_1: single_simulation(ENVIRONMENT, x_1), x)))


# NOTE using numba optimization, probabily not very effective
@njit
def single_crossover(x_1, x_2, alpha):
    return alpha * x_1 + (1 - alpha) * x_2


# NOTE using numba optimization here as well
@njit
def single_mutation(x_1, p_mutation, sigma_mutation):
    if np.random.random() < p_mutation:
        # FIXME for now doing the mutation on every part of the genes
        return np.clip(
            x_1 + np.random.normal(0, np.sqrt(sigma_mutation), len(x_1)), -1, 1
        )
    return x_1


@njit
def select_top_p(x, fitness, p):
    top_idx = np.argsort(fitness)[-p:][::-1]  # NOTE these are sorted
    return x[top_idx]


# FIXME is very untested
# FIXME for now we are just randomly selecting from the top 100
@njit
def selection_and_generation(x, fitness):
    x_new = np.zeros((POPULATION_SIZE, NUM_VARS))
    x_new[:100] = select_top_p(x, fitness, 100)  # Select top 100 nodes

    for i in range(100, 1000):
        # FIXME for now alpha is just 0.5, but likely we need to randomly generate this or something
        p = np.random.randint(0, 100)
        q = np.random.randint(0, 100)
        x_new[i] = single_crossover(x_new[p], x_new[q], 0.5)

        # FIXME for now fixed mutation rate and fixed sigma
        x_new[i] = single_mutation(x_new[i], 0.2, 0.1)

    return x_new


def init_params():
    return np.random.uniform(-1, -1, (POPULATION_SIZE, NUM_VARS))


def print_statistics(iter, fitness):
    print(
        f"Iteration {iter}: Average Fitness {np.mean(fitness)}, Best Fitness {np.max(fitness)}"
    )


# FIXME rewrite s.t. fitness is continuously recomputed
def run_simulation():
    iter = 0
    x = init_params()
    fitness = run_all_simulations(x)
    print_statistics(iter, fitness)
    x = selection_and_generation(x, fitness)

    for i in range(1, TRAINING_GENERATIONS):
        fitness = run_all_simulations(x)
        x = selection_and_generation(x, fitness)
        print_statistics(i, fitness)


if __name__ == "__main__":
    run_simulation()
