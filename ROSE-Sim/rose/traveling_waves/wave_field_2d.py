"""
2-D traveling-wave analysis placeholder.
"""
import pandas as pd

class TravellingWaveAnalyzer:
    def __init__(self, montage_path, data, sfreq, low_freq_band):
        self.sf = sfreq

    def compute_metrics(self):
        # return dummy DataFrame
        return pd.DataFrame({"metric": ["dummy"], "value": [0]})
