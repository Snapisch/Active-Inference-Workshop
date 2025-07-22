#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 15:02:16 2021

@author: sascha
"""

import numpy as np

def ex12_GenerativeModel(transition, observation, prior, timesteps):
    init_state_idx = np.random.choice(np.arange(1, len(prior)+1), p=prior)
    state = np.zeros(len(prior))
    state[init_state_idx-1] = 1
    obs = observation@state #Vector with obs probabilities
    
    actual_observation_idx = np.random.choice(np.arange(1, len(prior)+1), p=obs)
    actual_observation = np.zeros(len(prior))
    actual_observation[actual_observation_idx-1] = 1 # observation vector
    
    statehist = np.zeros((len(prior),timesteps+1))
    statehist[:,0] = state
    
    obshist = np.zeros((len(prior),timesteps+1))
    obshist[:,0] = actual_observation
    
    for t in range(timesteps):
        state = transition@state
        
        state_idx = np.random.choice(np.arange(1, len(state)+1), p=state)
        state = np.zeros(len(prior))
        state[state_idx-1] = 1
        statehist[:,t+1] = state    
        
        obs = observation@state

        actual_observation_idx = np.random.choice(np.arange(1, len(prior)+1), p=obs)
        actual_observation = np.zeros(len(prior))
        actual_observation[actual_observation_idx-1] = 1
        obshist[:, t+1] = actual_observation
        
    return statehist, obshist