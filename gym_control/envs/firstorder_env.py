#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the simplifie Banana selling environment.

Each episode is selling a single banana.
"""

# core modules
import logging.config
import math
import pkg_resources
from gym.utils import seeding

# 3rd party modules
from gym import spaces
import cfg_load
import gym
import numpy as np


path = 'config.yaml'  # always use slash in packages
filepath = pkg_resources.resource_filename('gym_control', path)
config = cfg_load.load(filepath)
logging.config.dictConfig(config['LOGGING'])


def get_chance(x):
    """Get probability that a banana will be sold at price x."""
    e = math.exp(1)
    return (1.0 + e) / (1. + math.exp(x + 1))


class FirstOrderEnv(gym.Env):
    """
    Define a simple Banana environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        logging.info("ControlEnv - Version {}".format(self.__version__))        

        self.viewer = None
        
        self.A = 0.9;
        self.B = 0.1;
        self.min_action = -10
        self.max_action = 10
        
        self.low_state = -10
        self.high_state = 10

        
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, shape=(1,), dtype=np.float32)
        
        self.seed()
        self.reset()


    def step(self, action):

        u = action
        
        # dynamic equations 
        x_old = self.state[0]
        x_new = self.A*x_old + self.B*u
        
        # reward dynamics
        reward = 0
        
        # done check
        done = False
        self.count = self.count + 1
        if self.count == 100:
            done = True
            
        self.state = np.array([x_new])
        return self.state, reward, done, {}
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def reset(self):
        self.state = np.array([0])
        self.count = 0
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = [self.TOTAL_TIME_STEPS - self.curr_step]
        return ob
