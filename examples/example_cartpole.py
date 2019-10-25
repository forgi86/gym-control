#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:06:01 2019

@author: marco
"""

import gym
import gym_control # contains the first order system environment
import matplotlib.pyplot as plt
import numpy as np

OBS = []
REW = []
ACT = []

if __name__ == '__main__':
    #env_firstorder = gym.make('FirstOrderControl-v0')
    env_cartpole = gym.make("CartPoleControl-v0")
    env = env_cartpole 
    
    total_reward = 0.0
    total_steps = 0
    obs = env.reset()
        
        
    while True:
        #action = env.action_space.sample()
        action = np.array(0)
        obs, reward, done, _ = env.step(action)
        OBS.append(obs[2])
        REW.append(reward)
        ACT.append(action)
        total_reward += reward
        total_steps += 1
        if done:
            break
        
# In[1]

    OBS = np.array(OBS)
    REW = np.array(REW)
    ACT = np.array(ACT)
    OBS_PI = ( OBS + np.pi) % (2 * np.pi ) - np.pi
    OBS_ERR = np.abs(OBS_PI)
    t = np.arange(0, len(OBS))*env.Ts

    fig,ax=plt.subplots()
    plt.plot(t,OBS)
    plt.plot(t,OBS_PI)
    plt.plot(t,OBS_ERR)
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)

    fig,ax=plt.subplots()
    plt.plot(t,OBS, 'k', label="theta")
    plt.plot(t,REW, 'r', label="reward")
    plt.xlabel("Time (s)")
    plt.ylabel("Angle (rad)")
    plt.grid(True)
    plt.legend()

    plt.show(block=False) # do not lock execution ("Run with python console" checkbox active)

