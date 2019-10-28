#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 14 10:06:01 2019

@author: marco
"""

import gym
import gym_control  # contains the first order system environment
import matplotlib.pyplot as plt
import numpy as np

OBS = []
REW = []
ACT = []
I = []

if __name__ == '__main__':
    env_arx = gym.make("ArxIdentification-v0", leaky='NO')
    env = env_arx

    total_reward = 0.0
    total_steps = 0
    obs_0 = env.reset() # s0
    reward_0 = 0.0

    OBS.append(obs_0) # s0
    REW.append(reward_0) #

    I.append(env.I)
    while True:
        #action = env.action_space.sample()
        action = 1.0 * np.random.randn(1)
        #action = np.array([1.0]) #
        obs, reward, done, _ = env.step(action)

        ACT.append(action) # Ai
        OBS.append(obs) # Si+1
        REW.append(reward) # Ri+1

        I.append(env.I)
        total_reward += reward
        total_steps += 1
        if done:
            break

    ACT.append(np.nan)
    G = np.cumsum(REW, axis=0)

    # In[1]

    OBS = np.vstack(OBS)
    REW = np.vstack(REW)
    ACT = np.vstack(ACT)
    I = np.stack(I)

    #I = np.vstack(I)
    t = np.arange(0, len(OBS))

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(t, OBS[:, 0])
    ax[0].set_xlabel("Time index (-)")
    ax[0].set_ylabel("State (-)")
    ax[0].grid(True)

    ax[1].plot(t, ACT)
    ax[1].set_xlabel("Time index (-)")
    ax[1].set_ylabel("Action (-)")
    ax[1].grid(True)

    ax[2].plot(t, REW)
    ax[2].set_xlabel("Time index (-)")
    ax[2].set_ylabel("Reward (-)")
    ax[2].grid(True)

    ax[3].plot(t, G)
    ax[3].set_xlabel("Time index (-)")
    ax[3].set_ylabel("Return (-)")
    ax[3].grid(True)



    OBS = np.vstack(OBS)
    REW = np.vstack(REW)
    ACT = np.vstack(ACT)
    I = np.stack(I)

    #I = np.vstack(I)
    t = np.arange(0, len(OBS))

    fig, ax = plt.subplots(4, 1)
    ax[0].plot(t, I[:, 0, 0])
    ax[0].set_xlabel("Time index (-)")
    ax[0].set_ylabel("$I_{11}$")
    ax[0].grid(True)

    ax[1].plot(t, I[:, 0, 1])
    ax[1].set_xlabel("Time index (-)")
    ax[1].set_ylabel("$I_{12}$")
    ax[1].grid(True)

    ax[2].plot(t, I[:, 1, 0])
    ax[2].set_xlabel("Time index (-)")
    ax[2].set_ylabel("$I_{21}$")
    ax[2].grid(True)

    ax[3].plot(t, I[:, 1,1])
    ax[3].set_xlabel("Time index (-)")
    ax[3].set_ylabel("$I_{22}$")
    ax[3].grid(True)
