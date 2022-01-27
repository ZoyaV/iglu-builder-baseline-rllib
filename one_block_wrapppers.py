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

class CompleteReward(gym.Wrapper):
    def check_complete(self, info):
        res = info['target_grid'] - info['grid']
        res[res < 0] = 0
        return len(np.where(res != 0)[0]) == 0

    def step(self, action):
        obs, reward, done, info = super().step(action)
        done = self.check_complete(self, info)
        if done:
            reward = 1
        else:
            reward = 0
        return obs, reward, done, info


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
