import gym
import numpy as np
from iglu.tasks import CustomTasks


def make_3d_cube():
    custom_grid = np.zeros((9, 11, 11)) # (y, x, z)
    custom_grid[0, 5, 5] = 1 # blue color
    return CustomTasks([
        ('<Architect> Please, build a stack of three blue blocks somewhere.\n'
         '<Builder> Sure.',
         custom_grid)])
