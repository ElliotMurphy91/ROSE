"""
Toy RNN that outputs random state vectors; real model TBD.
"""
import numpy as np

class MotifRNN:
    @classmethod
    def load_pretrained(cls):
        return cls()

    def run_transcript(self, words):
        # shape: (time, dim)
        rng = np.random.default_rng(1)
        return rng.standard_normal((len(words), 16))
