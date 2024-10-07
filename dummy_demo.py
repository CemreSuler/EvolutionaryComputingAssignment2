################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy as np
import time

from tqdm import trange

from evoman.environment import Environment

experiment_name = "dummy_demo"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(
    experiment_name=experiment_name,
    enemies=[1, 2, 3, 4, 5, 6, 7, 8],
    visuals=False,
    speed="fastest",
    multiplemode="yes",
)


# NOTE change fitness by changing
env.fitness_single
env.cons_multi = lambda values: values  # Ensure that all values are returned
# v, _, _, _ = env.play()
a = np.array([[1, 2, 3], [4, 5, 6]])
print(a[0])

t = time.time()
for i in trange(1000):
    env.play()
print(time.time() - t)
