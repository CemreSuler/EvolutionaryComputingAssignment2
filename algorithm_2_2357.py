# NOTE importing Numba to make sure that important calculations happen much faster
from joblib import Memory
from numba import njit

# from tqdm import trange

from evoman.environment import Environment

# NOTE renamed to adhere to PIP conventions
from demo_controller import player_controller as PlayerController

import pandas as pd
import numpy as np

import time
import os

## SETTINGS
# PyGame Settings
HEADLESS = True
EXPERIMENT_NAME = "Algorithm2_2357"

# Model settings
NUM_HIDDEN_NEURONS = 10
POPULATION_SIZE = 50  # NOTE population size per island
TRAINING_GENERATIONS = 50
SWITCH_ITERATIONS = 10
MUTATION_RATE = 0.2
SELECT_TOP = 10  # Selects the top to create offspring, kill this part
ISLANDS = 5

# Alpha of combination
MIN_ALPHA = 0.1
MAX_ALPHA = 0.9

# Matrix allowed values
MIN_MATRIX = -1
MAX_MATRIX = 1
## END OF SETTINGS


if HEADLESS:
    os.environ["SDL_VIDEODRIVER"] = "dummy"

# NOTE contact_hurt already set as default
ENVIRONMENT = Environment(
    experiment_name=EXPERIMENT_NAME,
    enemies=[2, 3, 5, 7],
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
# ENVIRONMENT.cons_multi = lambda values: values
NUM_VARS = (ENVIRONMENT.get_num_sensors() + 1) * NUM_HIDDEN_NEURONS + (
    NUM_HIDDEN_NEURONS + 1
) * 5

memory = Memory("./.cache", verbose=0)


# FIXME need to create a function that combines the different values in some sort of good way, for now just taking the mean
# NOTE numba optimization not possible
@memory.cache
def single_simulation(x_1):
    fitness, player_life, enemy_life, time = ENVIRONMENT.play(pcont=x_1)
    # return float(np.mean(fitness))
    return fitness


# NOTE numba optimization not possible
def run_all_simulations(x):
    return np.array(list(map(lambda x_1: single_simulation(x_1), x)))


# NOTE using numba optimization, probabily not very effective
@njit
def single_crossover(x_1, x_2, alpha):
    return alpha * x_1 + (1 - alpha) * x_2


# NOTE using numba optimization here as well
@njit
def single_mutation(x_1, p_mutation, sigma_mutation):
    r_mutation_prob = np.random.random(len(x_1))  # Generate probability of mutation
    r_mutation_prob[r_mutation_prob > p_mutation] = 0  # If prob too low, then set zero
    r_mutation_value = np.random.normal(
        0, np.sqrt(sigma_mutation), len(x_1)
    )  # Generate mutation value

    # Perform mutation and clip to ensure that remains in right value range
    return np.clip(
        np.ceil(r_mutation_prob) * r_mutation_value,
        MIN_MATRIX,
        MAX_MATRIX,
    )


@njit
def select_top_p(x, fitness, p):
    top_idx = np.argsort(fitness)[-p:][::-1]  # NOTE these are sorted
    return x[top_idx]


@njit
def compute_sigma(fitness):
    return 1 / (5 * max(1, np.max(fitness) / 10))


# FIXME for now we are just randomly selecting from the top 100
@njit
def selection_and_generation(x, fitness):
    x_new = np.zeros((POPULATION_SIZE * ISLANDS, NUM_VARS))

    for i in range(ISLANDS):
        x_new[
            i * POPULATION_SIZE : i * POPULATION_SIZE + POPULATION_SIZE - SELECT_TOP
        ] = select_top_p(
            x[i * POPULATION_SIZE : (i + 1) * POPULATION_SIZE],
            fitness[i * POPULATION_SIZE : (i + 1) * POPULATION_SIZE],
            POPULATION_SIZE - SELECT_TOP,
        )
        sigma_i = compute_sigma(fitness)

        for j in range(POPULATION_SIZE - SELECT_TOP, POPULATION_SIZE):
            p = np.random.randint(0, SELECT_TOP)
            q = np.random.randint(0, SELECT_TOP)
            alpha = np.random.uniform(MIN_ALPHA, MAX_ALPHA)
            x_new[POPULATION_SIZE * i + j] = single_crossover(
                x_new[POPULATION_SIZE * i + p], x_new[POPULATION_SIZE * i + q], alpha
            )

            x_new[POPULATION_SIZE * i + j] = single_mutation(
                x_new[POPULATION_SIZE * i + j], MUTATION_RATE, sigma_i
            )

    return x_new


@njit
def switch_best(x_old):
    x_best = x_old[::POPULATION_SIZE]
    x_new = x_old.copy()

    for i in range(ISLANDS):
        j = i + 1
        if j >= ISLANDS:
            j = 0
        x_new[i * POPULATION_SIZE] = x_best[j]

    return x_new


def init_params():
    return np.random.uniform(
        MIN_MATRIX, MAX_MATRIX, (POPULATION_SIZE * ISLANDS, NUM_VARS)
    )


def print_statistics(iter, fitness):
    print(
        f"1: {np.max(fitness[:50])}, 2: {np.max(fitness[50:100])}, 3: {np.max(fitness[100:150])}, 4: {np.max(fitness[150:200])}, 5: {np.max(fitness[200:])}"
    )

# FIXME rewrite s.t. fitness is continuously recomputed
def run_simulation():
    results = [("Best", "Mean", "Best X")]
    x = init_params()
    fitness = run_all_simulations(x)
    print_statistics(0, fitness)
    results.append((np.max(fitness), np.mean(fitness), x[np.argmax(fitness)]))
    x = selection_and_generation(x, fitness)

    for i in range(1, TRAINING_GENERATIONS):
        fitness = run_all_simulations(x)
        results.append((np.max(fitness), np.mean(fitness), x[np.argmax(fitness)]))
        x = selection_and_generation(x, fitness)
        print_statistics(i, fitness)

        if i % 10 == 0:
            x = switch_best(x)

    os.makedirs(f"data/{EXPERIMENT_NAME}", exist_ok=True)
    pd.DataFrame(results).to_csv(
        f"data/{EXPERIMENT_NAME}/{int(time.time())}-{os.getpid()}.csv"
    )
    os.makedirs(f"data/{EXPERIMENT_NAME}/competition", exist_ok=True)
    np.savetxt(f"data/{EXPERIMENT_NAME}/competition/{int(time.time())}-{os.getpid()}-best_individual.txt", x[np.argmax(fitness)])

if __name__ == "__main__":
    run_simulation()
