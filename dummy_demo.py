################################
# EvoMan FrameWork - V1.0 2016 #
# Author: Karine Miras         #
# karine.smiras@gmail.com      #
################################

# imports framework
import sys, os
import numpy as np

from evoman.environment import Environment

experiment_name = "dummy_demo"
if not os.path.exists(experiment_name):
    os.makedirs(experiment_name)

# initializes environment with ai player using random controller, playing against static enemy
env = Environment(
    experiment_name=experiment_name,
    enemies=[1, 2, 3, 4, 5],
    visuals=False,
    speed="fastest",
    multiplemode="yes",
)


# NOTE change fitness by changing
env.fitness_single
env.cons_multi = lambda values: values  # Ensure that all values are returned
v, _, _, _ = env.play()
print(v)
print("a")
print(np.arcsinh(100))
print(np.arcsinh(10))
print(np.arcsinh(0))
print(np.arcsinh(-10))
print(np.arcsinh(-100))
print("b")
print(v.mean())
print(np.mean(np.arcsinh(v)))
print(env.get_playerlife())
