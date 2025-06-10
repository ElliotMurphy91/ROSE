"""
1-D travelling δ-wave controls workspace depth (E-level).
"""

import numpy as np, matplotlib.pyplot as plt
fs, dur = 500, 4
x   = np.linspace(0, 1, 40)                    # cortical axis
t   = np.arange(0, dur, 1/fs)
δf  = 2
capacity = 4

# wave: phase = 2π(δft - kx)
k = 4                                          # spatial freq
δfield = np.sin(2*np.pi*δf*t[:,None] - 2*np.pi*k*x)

workspace = []
alpha = np.zeros_like(t)

for ti,phi in enumerate(δfield):
    # posterior (x≈0) trough triggers 'push'
    if phi[0] < -0.99:
        workspace.append('chunk')
        if len(workspace) > capacity:
            workspace.pop(0)
        alpha[ti:] = len(workspace)/capacity   # inhibitory α load

print("Final workspace depth:", len(workspace))

plt.figure(figsize=(9,3))
plt.subplot(121)
plt.imshow(δfield.T, aspect='auto', origin='lower',
           extent=[0,dur,0,1], cmap='RdBu_r')
plt.xlabel('time (s)'); plt.ylabel('cortex →')
plt.title('δ travelling wave')
plt.subplot(122)
plt.plot(t, alpha, 'k')
plt.ylim(0,1.1); plt.xlabel('time (s)')
plt.title('α-load (workspace usage)')
plt.tight_layout(); plt.show()
