
from rose.analysis import modulation_index
import numpy as np
def test_mi_zero_on_noise():
    rng=np.random.default_rng(0)
    sig=rng.standard_normal(1000)
    mi=modulation_index(sig,sig)
    assert mi<0.05
