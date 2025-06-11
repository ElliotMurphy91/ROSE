"""
Simple ephaptic-field demo returning zeros.
"""
import numpy as np

class EphapticField:
    def __init__(self, num_neurons=100, field_radius=4.0):
        self.N = num_neurons

    def simulate(self, duration=1.0, dt=1e-3):
        steps = int(duration / dt)
        return np.zeros((steps, self.N))
