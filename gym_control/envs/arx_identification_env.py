#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the simplifie Banana selling environment.

Each episode is selling a single banana.
"""

# core modules
import logging.config
import numpy as np
import pkg_resources

# 3rd party modules
from gym.utils import seeding
from gym import spaces
import cfg_load
import gym

path = 'config.yaml'  # always use slash in packages
filepath = pkg_resources.resource_filename('gym_control', path)
config = cfg_load.load(filepath)
logging.config.dictConfig(config['LOGGING'])


def chol_encode(SIG):
    n_par = SIG.shape[0]
    L = np.linalg.cholesky(SIG)
    theta_L = L[np.tril_indices(n_par)]
    return theta_L


def chol_decode(theta_L, n_par):
    L = np.zeros((n_par, n_par))
    L[np.tril_indices(n_par)] = theta_L
    SIG = L @ np.transpose(L)
    return SIG


def logchol_encode(SIG):
    n_par = SIG.shape[0]
    L = np.linalg.cholesky(SIG)
    theta_L = L[np.tril_indices(n_par)]
    idx_diag = np.arange(n_par)
    idx_diag = (idx_diag + 1) * (idx_diag + 2) // 2 - 1
    theta_L[idx_diag] = np.log(theta_L[idx_diag])
    return theta_L


def logchol_decode(theta_L, n_par):
    idx_diag = np.arange(n_par)
    idx_diag = (idx_diag + 1) * (idx_diag + 2) // 2 - 1
    theta_L[idx_diag] = np.exp(theta_L[idx_diag])
    L = np.zeros((n_par, n_par))
    L[np.tril_indices(n_par)] = theta_L
    SIG = L @ np.transpose(L)
    return SIG

class ArxIdentificationEnv(gym.Env):
    """
    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self, leaky='NO', episode_len=300):
        self.__version__ = "0.1.0"
        logging.info("ControlEnv - Version {}".format(self.__version__))

        self.viewer = None

        self.A = 0.9
        self.B = 0.1

        self.std_noise = 0.03 #  noise standard deviation
        self.std_x0 = 0.3
        self.S0 = np.diag([0.01, 0.01]) # parameter random walk covariance

        self.min_action = -1
        self.max_action =  1

        self.low_state = -2
        self.high_state = 2

        self.leaky = leaky
        self.episode_len = episode_len

        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, shape=(4,), dtype=np.float32)

        self.seed()
        self.reset()

    def step(self, action):
        u = action

        # dynamic equations
        x_old = self.sys_state
        x_new = self.A * x_old + self.B * u + self.std_noise*np.random.randn()

        phi_t = np.array([x_new, u]).reshape(1, -1)
        I_t = phi_t * np.transpose(phi_t)/(self.std_noise**2)

        if self.leaky == 'KAL': #
            I_new = np.linalg.inv(np.linalg.inv(self.I + I_t) + self.S0)
        elif self.leaky == 'RLS':
            I_new =  0.9*np.copy(self.I) + I_t
        elif self.leaky == 'NO':
            I_new = self.I + I_t
        else:
            raise ValueError("Wrong option for leaky parameter!")

        reward = np.min(np.linalg.eigvals(I_new)) - np.min(np.linalg.eigvals(self.I)) # reward is the increase of the smallest eigenvalue of I
        reward = reward/1000

        self.I = I_new
        self.sys_state = x_new

        self.I_state = logchol_encode(self.I) #self.sys_state
        self.env_state[0] = self.sys_state
        self.env_state[1:] = self.I_state

#        self.env_state[1] = self.I[0, 0]
#        self.env_state[2] = self.I[0, 1]
#        self.env_state[3] = self.I[1, 1]

        # done check
        done = False

        self.count = self.count + 1
        if self.count == self.episode_len:
            done = True

        return np.copy(self.env_state), reward, done, {}

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def reset(self):
        self.sys_state = self.std_x0*np.random.randn(1) # initial system state
        self.sys_state = np.clip(self.sys_state, self.low_state, self.high_state)
        self.I0 = np.diag([1.0, 1.0]) # initial information matrix
        self.I = np.copy(self.I0)
        self.I_state = logchol_encode(self.I)

        self.env_state = np.zeros(4)
        self.env_state[0] = self.sys_state
        self.env_state[1:] = self.I_state

        self.count = 0 # ?

        return np.copy(self.env_state)

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = self.env_state
        return ob
