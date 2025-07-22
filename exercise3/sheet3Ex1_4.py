#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 13:57:51 2021

@author: sascha
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solution to Sheet 2 Exercise 1.2 (Grid World Generative Model)

@author: Sascha Frölich
"""

import numpy as np
import matplotlib.pyplot as plt
from ex12_GenerativeModel import ex12_GenerativeModel

np.random.seed(226)

def ex14_BeliefInferenceMF(transition_matrix, observation_matrix, prior, observations):
    
    "For now without Backward Inference"
    size_statespace = observations.shape[0]
    timesteps = observations.shape[1]
    beliefs = np.zeros((size_statespace, timesteps))

    "Set the believe of s[0] to the prior"
    beliefs[:, 0] = prior

    trans = transition_matrix.copy()
    obs = observation_matrix.copy()
    "With exp(a*log(b)) = b**a"
    for t in range(1, timesteps):
        for row in range(size_statespace):
            beliefs[row, t] = 1
            for col in range(size_statespace):
                if trans[row, col] == 0:
                    trans[row, col] = 10**-10
                if obs[col, row] == 0:
                    obs[col, row] = 10**-10
                
                beliefs[row, t] *= (trans[row, col]**beliefs[col, t-1]) *\
                                    (obs[col, row]**observations[col, t-1])
                                    
        if beliefs[:, t].sum() == 0:
            beliefs[:, t] = 1/beliefs[:, t].size
        
        beliefs[:, t] = beliefs[:, t]/ beliefs[:, t].sum()
        
         
    ## Don't return prior    
    return beliefs[:, 1:]


#%% Define prior, transition matrix and observation matrix
prior = np.array([0,0,0,0,0,0,0,0,0.25,0.25,0,0,0.25,0.25,0,0])

"""Create Observations for later inference with Generative Model """

transition_Mtx = np.array([
[0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,0. , 0. , 0. ],
[1. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. , 0. ,0. , 0. , 0. ],
[0. , 1. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. , 0. , 0. ,0. , 0. , 0. ],
[0. , 0. , 1. , 1. , 0. , 0. , 0. , 1. , 0. , 0. , 0. , 0. , 0. ,0. , 0. , 0. ],
[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. ,0. , 0. , 0. ],
[0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. ,0. , 0. , 0. ],
[0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. , 0.5, 0. , 0. ,0. , 0. , 0. ],
[0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. , 1. , 0. ,0. , 0. , 0. ],
[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5,0. , 0. , 0. ],
[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. , 0. ,0.5, 0. , 0. ],
[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. , 0. ,0. , 0.5, 0. ],
[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. ,0. , 0. , 1],
[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,0. , 0. , 0. ],
[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5,0. , 0. , 0. ],
[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,0.5, 0. , 0. ],
[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. ,0. , 0.5, 0. ]])

observation_Mtx = np.array([
[0.5, 0.16666667, 0., 0., 0.16666667,0., 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
[0.25, 0.5, 0.16666667, 0.,0.,0.125, 0.,0.,0.,0.,0.,0.,0.,0.,0.,0.],
[0.,0.16666667,0.5,0.25,0.,0.,0.125,0.,0.,0.,0.,0.,0.,0.,0.,0.],
[0.,0.,0.16666667, 0.5, 0.,0.,0., 0.16666667,0.,0.,0.,0.,0.,0.,0.,0.],
[0.25, 0.,0.,0., 0.5,0.125,0.,0., 0.16666667, 0.,0.,0.,0.,0.,0.,0.],
[0., 0.16666667, 0.,0., 0.16666667,0.5, 0.125, 0.,0., 0.125,0.,0.,0.,0.,0.,0.],
[0.,0.,0.16666667,0.,0.,0.125, 0.5, 0.16666667,0.,0.,0.125, 0.,0.,0.,0.,0.],
[0.,0.,0.,0.25,0.,0.,0.125, 0.5,0.,0.,0.,0.16666667,0.,0.,0.,0.],
[0.,0.,0.,0., 0.16666667,0.,0.,0., 0.5, 0.125,0.,0., 0.25, 0.,0.,0.],
[0.,0.,0.,0.,0.,0.125,0.,0.,0.16666667, 0.5,0.125,0.,0.,0.16666667,0.,0.],
[0.,0.,0.,0.,0.,0.,0.125, 0.,0., 0.125,0.5, 0.16666667, 0.,0., 0.16666667,0.],
[0.,0.,0.,0.,0.,0.,0.,0.16666667, 0.,0.,0.125, 0.5, 0.,0.,0.,0.25],
[0.,0.,0.,0.,0.,0.,0.,0.,0.16666667, 0.,0.,0.,0.5, 0.16666667, 0.,0.],
[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.125,0.,0.,0.25, 0.5, 0.16666667,0.],
[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.125,0.,0.,0.16666667, 0.5,0.25],
[0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.,0.16666667,0.,0.,0.16666667,0.5]])

#%% Generate Observations and perform Inference

hist, obshist = ex12_GenerativeModel(transition_Mtx,observation_Mtx,prior,6)

no_observations = obshist.shape[1]

"""Perform Inference based on observations"""
# Infer states
inferred_states = ex14_BeliefInferenceMF(transition_Mtx, observation_Mtx, prior, obshist)

#%% 

for i in range(inferred_states.shape[1]):
    plt.imshow(obshist[:, i].reshape(4,4), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Observation %d'%i)
    #plt.savefig('figures/inference%d'%i)
    plt.show()
    
    plt.imshow(inferred_states[:, i].reshape(4,4), vmin=0, vmax=1)
    plt.colorbar()
    plt.title('Inferred State %d'%i)
    #plt.savefig('figures/inference%d'%i)
    plt.show()