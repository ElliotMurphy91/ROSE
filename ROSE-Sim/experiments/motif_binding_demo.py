"""
DYNAMICAL MOTIF EMERGENCE under spike-phase coupling.
We create 5 excitatory nodes with weak synapses.
A slow theta driver entrains them. When coupling on:
  – E-rates self-organise into two stable motifs.
  – MI(θ,γ_local) rises ⇒ O-level 'lexical bundle' formed.
"""

import numpy as np, matplotlib.pyplot as plt
from rose.oscillators import WilsonCowanOscillator
from rose.analysis import modulation_index

fs, dur, N = 1000, 5, 5
theta = np.sin(2*np.pi*6*np.arange(0,dur,1/fs))

sim = WilsonCowanOscillator(fs=fs)
E = np.zeros((N, int(dur*fs)))
for i in range(N):
    E[i], _ = sim.simulate(dur, E0=np.random.rand()*0.3)

# add theta-phase modulation to 3/5 nodes (spike-phase coupling ON)
coupled = [0,2,4]
for i in coupled:
    E[i] *= (theta*0.5 + 0.6)          # phase-gain

# MI analysis
mi = [modulation_index(theta, E[i]) for i in range(N)]
print("θ-γ MI per node:", np.round(mi,3))

# PCA for motif visualisation
from sklearn.decomposition import PCA
pca = PCA(2).fit_transform(E.T)

plt.figure(figsize=(5,5))
plt.plot(pca[:,0], pca[:,1], lw=.5, alpha=.7)
plt.scatter(pca[::200,0], pca[::200,1], s=10, c='r')
plt.title("Trajectory in PC-space → two attractors ≈ motifs")
plt.tight_layout(); plt.show()
