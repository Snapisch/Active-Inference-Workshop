#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Inference Function. For observations up to time-point t, infer posterior over states
for time-point t.

Inference is achieved by Computing p(s_t,o_{1:t}^{obs}) and normalizing this distribution
at the end.

@author: Sascha Frölich
"""

import numpy as np
import matplotlib.pyplot as plt
from sheet2Ex1_2 import ex12_GenerativeModel

np.random.seed(226)

def infer(transition_matrix,observation_matrix,prior,obshist):
    # Infer state p(s_t | o_1:t)
    no_obs = obshist.shape[1]
    
    state = prior
    
    # compute state = auxiliary prior
    for t in range(no_obs-1):
        # Compute sum over all states 1:t-1
        observed_state = np.where(obshist[:,t]==1)[0][0]
        
        SUM = 0
        for si in range(state.size): # Compute sum of auxiliary prior.
            # Mach als .sum()
            SUM += observation_matrix[observed_state,si]*state[si]*transition_matrix[:,si]
        
        state = SUM.copy()
    
    # last observation
    observed_state = np.where(obshist[:,-1]==1)[0][0]
    
    p_st = observation_matrix[observed_state,:]*state
    
    p_st = p_st / p_st.sum() # Normalize
    
    return p_st

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

# This should sum to 1 for each row: observation.sum(axis=0)

prior = np.array([0,0,0,0,0,0,0,0,0.25,0.25,0,0,0.25,0.25,0,0])

"""Create Observations for later inference with Generative Model """

hist, obshist = ex12_GenerativeModel(transition_Mtx,observation_Mtx,prior,6)

no_observations = obshist.shape[1]

class InferenceNavigator:
    """
    An interactive plot navigator for a sequence of inferred states.
    Use left and right arrow keys to navigate.
    """
    def __init__(self, transition_Mtx, observation_Mtx, prior, obshist):
        self.transition_Mtx = transition_Mtx
        self.observation_Mtx = observation_Mtx
        self.prior = prior
        self.obshist = obshist
        self.no_observations = obshist.shape[1]
        self.current_index = 0
        
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.im = None
        self.cbar = None
        
        self.update_plot()
        print("Showing interactive plot. Use left/right arrow keys to navigate through inferred states.")
        plt.show()

    def on_key(self, event):
        """Handles key press events for navigation."""
        if event.key == 'right':
            self.current_index = (self.current_index + 1) % self.no_observations
        elif event.key == 'left':
            self.current_index = (self.current_index - 1 + self.no_observations) % self.no_observations
        else:
            return
        self.update_plot()

    def update_plot(self):
        """Updates the plot to show the current inferred state."""
        inferred_state = infer(self.transition_Mtx, self.observation_Mtx, self.prior, self.obshist[:, :self.current_index + 1])
        
        if self.im is None:
            self.im = self.ax.imshow(inferred_state.reshape(4, 4), vmin=0, vmax=1)
            self.cbar = self.fig.colorbar(self.im, ax=self.ax)
        else:
            self.im.set_data(inferred_state.reshape(4, 4))
        
        self.ax.set_title(f'Inferred State at t={self.current_index + 1}/{self.no_observations}')
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.draw()

# Create an instance of the interactive plot navigator
InferenceNavigator(transition_Mtx, observation_Mtx, prior, obshist)