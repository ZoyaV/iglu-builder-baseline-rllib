import gym
import numpy as np
from iglu.tasks import CustomTasks
import random

def make_3d_cube(rand = False):
    custom_grid = np.zeros((9, 11, 11)) # (y, x, z)
    if rand:
        x = random.randint(0,10)
        y = random.randint(0,10)
        #z = random.randint(0, 8)
    else:
        x = 5
        y = 5
    z = 0
    print("\n CHANGE TASK \n", x,y,z)
    print(z,x,y)
    custom_grid[z, x, y] = 1 # blue color
    return CustomTasks([
        ('<Architect> Please, build a stack of three blue blocks somewhere.\n'
         '<Builder> Sure.',
         custom_grid)])
