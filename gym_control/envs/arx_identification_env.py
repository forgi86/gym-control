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



class ArxIdentificationEnv(gym.Env):
    """
    Define a simple Banana environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, leaky='NO'):
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

        self.leaky = leaky

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, shape=(4,), dtype=np.float32)

        self.seed()
        self.reset()

    def step(self, action):
        u = action

        # dynamic equations
        x_old = self.sys_state
        x_new = self.A * x_old + self.B * u + self.std_noise*np.random.randn()

        phi_t = np.array([x_new, u]).reshape(1,-1)
        I_t = phi_t * np.transpose(phi_t)

        if self.leaky == 'KAL':
            I_new = np.linalg.inv(np.linalg.inv(self.I + I_t) + self.S0)
        elif self.leaky == 'RLS':
            I_new =  0.9*np.copy(self.I) + I_t
        elif self.leaky == 'NO':
            I_new = self.I + I_t
        else:
            raise ValueError("Wrong option for leaky parameter!")

        reward = np.min(np.linalg.eigvals(I_new)) - np.min(np.linalg.eigvals(self.I))

        self.I = I_new
        self.sys_state = x_new

        self.env_state[0] = self.sys_state
        self.env_state[1] = self.I[0, 0]
        self.env_state[2] = self.I[0, 1]
        self.env_state[3] = self.I[1, 1]

        # done check
        done = False

        self.count = self.count + 1
        if self.count == 100:
            done = True

        return np.copy(self.env_state), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.sys_state = np.random.randn(1) # initial system state
        self.I0 = np.diag([1.0, 1.0]) # initial information matrix
        self.I = np.copy(self.I0)

        self.env_state = np.empty(4)
        self.env_state[0] = self.sys_state
        self.env_state[1] = self.I[0, 0]
        self.env_state[2] = self.I[0, 1]
        self.env_state[3] = self.I[1, 1]

        self.count = 0 # ?
        return np.copy(self.env_state)

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = self.env_state
        return ob
