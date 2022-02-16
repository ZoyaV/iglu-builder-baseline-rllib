from threading import stack_size
import gym
import os
import cv2
import shutil
import datetime
import pickle
import json
import uuid
import logging
from gym.core import ActionWrapper
import numpy as np
from collections import defaultdict
from typing import Generator
from minerl_patched.herobraine.hero import spaces
from wrappers import Wrapper
from custom_tasks import make_3d_cube
from iglu.tasks import RandomTasks

class RandomTarget(gym.Wrapper):
    #current_env  = [[None]]
    def __init__(self, env, thresh = 0.37):
        super().__init__(env)
        self.thresh = thresh
        self.total_reward = 0
        self.sum = self.thresh/10
        self.count = 0
        self.changes = 0

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        if done:
            self.count += 1
            self.sum += reward
            self.total_reward = self.sum/ self.count
            # print("\n --- upd reward ---- \n")
            # print("blocks count = ", len(np.where(info['grid']!=0)[0]))
            # print("reward = ",self.total_reward)
            # print("sum = ", self.sum)
            # print("count = ", self.count)
            # print(" --- upd reward ---- \n")

            if (self.total_reward > self.thresh):

                # print("\n ----Make new Goal ----- \n")
                self.changes += 1
                task = RandomTasks(
                           max_blocks=1,
                           height_levels=1,
                           allow_float= False,
                           max_dist= 1,
                           num_colors= 1,
                           max_cache=10,
                        )
                self.update_taskset(task)
                self.sum = self.thresh / 10
                self.count = 0
                self.total_reward = self.thresh / 10
                info['new_env'] = True
        return  obs, reward, done, info



class SizeLongReward(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.count_actions = 0

    def check_complete(self, info):
        res = info['target_grid'] - info['grid']
        res[res < 0] = 0
        return len(np.where(res != 0)[0]) == 0

    def get_reward(self, info):
        res = info['target_grid'] - info['grid']
        res[res < 0] = 0
        return len(np.where(res != 0)[0])

    def step(self, action):
        self.count_actions += 1
        obs, reward, done, info = super().step(action)
        done = self.check_complete(self, info)
        reward = self.get_reward(info)/self.count_actions
        return obs, reward*100, done, info