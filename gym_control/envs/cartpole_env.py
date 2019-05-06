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

COEFF = {
        'M': 0.5,
        'm': 0.2,
        'b': 0.1,
        'ftheta': 0.1,
        'l': 0.3,
        'g': 9.81}

P_MAX = 10
P_MIN = -P_MAX
V_MAX = 100
V_MIN = - V_MAX
THETA_MAX = 2*np.pi
THETA_MIN = 0
OMEGA_MAX = 100
OMEGA_MIN = - OMEGA_MAX
F_MAX = 10
F_MIN = -F_MAX

TS = 1e-3

def wrap_to_pi(angle_2pi):

    angle_pi = (angle_2pi + np.pi) % (2 * np.pi ) - np.pi
    return angle_pi

class CartPoleEnv(gym.Env):
    """
    Define a simple Banana environment.

    The environment defines which actions can be taken at which point and
    when the agent receives which reward.
    """

    def __init__(self):
        self.__version__ = "0.1.0"
        logging.info("CartPoleEnv - Version {}".format(self.__version__))        

        self.Ts = TS
        self.min_action = F_MIN
        self.max_action = F_MAX
        self.low_state = np.array([P_MIN, V_MIN, THETA_MIN, OMEGA_MIN])
        self.high_state = np.array([P_MAX, V_MAX, THETA_MAX, OMEGA_MAX])
        self.viewer = None
        
        self.action_space = spaces.Box(low=self.min_action, high=self.max_action, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=self.low_state, high=self.high_state, dtype=np.float32)
        
        self.seed()
        self.reset()


    def step(self, action):

        # Coefficients
        M = COEFF['M']
        m = COEFF['m']
        b = COEFF['b']
        ftheta = COEFF['ftheta']
        l = COEFF['l']
        g = COEFF['g']
        
        F = action # input force
        x_old = self.state
        
        # assign states by name 
        #p = x_old[0]
        v = x_old[1]
        theta = x_old[2]
        omega = x_old[3]
        
        # Derivative computation
        der = np.zeros(np.shape(self.state)) 
        
        der[0] = v
        der[1] = (m*l*np.sin(theta)*omega**2 -m*g*np.sin(theta)*np.cos(theta)  + m*ftheta*np.cos(theta)*omega + F - b*v)/(M+m*(1-np.cos(theta)**2));
        der[2] = omega
        der[3] = ((M+m)*(g*np.sin(theta) - ftheta*omega) - m*l*omega**2*np.sin(theta)*np.cos(theta) -(F-b*v)*np.cos(theta))/(l*(M + m*(1-np.cos(theta)**2)) );
        
        # Forward euler step
   
        x_new = x_old + der*TS
        
        # reward dynamics
        reward = np.pi - np.abs(wrap_to_pi(theta)) # pi - abs of the angle in range (-pi, pi)
        
        # done check
        done = False
        self.count = self.count + 1
        if self.count == 20000:
            done = True
            
        self.state = np.array(x_new)
        return self.state, reward, done, {}
        
    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]
    

    def reset(self):
        self.state = np.array([0, 0, np.pi/2, 0])
        self.count = 0
        return np.array(self.state)

    def _render(self, mode='human', close=False):
        return

    def _get_state(self):
        """Get the observation."""
        ob = [self.TOTAL_TIME_STEPS - self.curr_step]
        return ob
