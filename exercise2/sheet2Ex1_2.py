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
[0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0. , 0.5, 0. , 0. ,0. , 0. , 1  ],
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

hist, obshist = ex12_GenerativeModel(transition_Mtx,observation_Mtx,prior,6)

class PlotNavigator:
    """
    An interactive plot navigator for a sequence of images.
    Use left and right arrow keys to navigate.
    """
    def __init__(self, obshist):
        self.obshist = obshist
        self.no_observations = obshist.shape[1]
        self.current_index = 0
        
        self.fig, self.ax = plt.subplots()
        self.fig.canvas.mpl_connect('key_press_event', self.on_key)
        
        self.update_plot()
        print("Showing interactive plot. Use left/right arrow keys to navigate through observations.")
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
        """Updates the plot to show the current observation."""
        self.ax.clear()
        self.ax.set_title(f'Observed State {self.current_index + 1}/{self.no_observations}')
        self.ax.imshow(self.obshist[:, self.current_index].reshape(4,4))
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.fig.canvas.draw()

# Create an instance of the interactive plot navigator
PlotNavigator(obshist)