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


class ArxIdentificationEnv(gym.Env):
    """
    Define a simple Banana environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        logging.info("ControlEnv - Version {}".format(self.__version__))

        self.viewer = None

        self.A = 0.9
        self.B = 0.1

        self.std_noise = 1.0 #  noise standard deviation
        self.S0 = np.diag([0.01, 0.01]) # parameter random walk covariance

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
        x_new = self.A * x_old + self.B * u + self.std_noise*np.random.randn()

        phi_t = np.array([x_new, u]).reshape(1,-1)
        I_t = phi_t * np.transpose(phi_t)

        #I_new = 0.9*np.copy(self.I) + I_t
        I_new = np.linalg.inv(np.linalg.inv(self.I + I_t) + self.S0)

        #reward = np.log(np.linalg.det(I_new)) - np.log(np.linalg.det(self.I))
        reward = np.min(np.linalg.eigvals(I_new)) - np.min(np.linalg.eigvals(self.I))

        #self.I = I_new
        self.I = I_new#np.linalg.inv(np.linalg.inv(I_new) + self.S0) # random walk in the parameter space

        self.state = np.array([x_new]) # also info matrix is a state!

        # done check
        done = False
        self.count = self.count + 1
        if self.count == 1000:
            done = True

        return self.state, reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.state = np.random.randn(1) # initial system state
        self.I0 = np.diag([1.0, 1.0]) # initial information matrix
        self.I = np.copy(self.I0)
        self.count = 0 # ?
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = self.x
        return ob
