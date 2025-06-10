
"""Operation-level coupling demo for ROSE.
   • theta–gamma PAC  → lexical feature-bundle assembly
   • delta–gamma PAC  → generic combinatorial binding (domain-general)
"""
import numpy as np, matplotlib.pyplot as plt
from rose.analysis import modulation_index

fs=1000; dur=3.0
t=np.arange(0,dur,1/fs)

delta=np.sin(2*np.pi*2*t)        # 2 Hz
theta=np.sin(2*np.pi*6*t)        # 6 Hz
gamma_carrier=np.sin(2*np.pi*80*t)

relu=lambda x: np.maximum(x,0.0)

# Lexical gamma: amplitude gated by theta ↑
gamma_theta = relu(theta) * gamma_carrier

# Generic combinatorial gamma: amplitude gated by delta ↑
gamma_delta = relu(delta) * gamma_carrier

mi_theta=modulation_index(theta,gamma_theta)
mi_delta=modulation_index(delta,gamma_delta)

print(f"Theta–gamma MI (lexical):  {mi_theta:.3f}")
print(f"Delta–gamma MI (generic):  {mi_delta:.3f}")

# --------- plot -------------
fig,ax=plt.subplots(2,1,figsize=(9,6),sharex=True)
ax[0].plot(t,theta,label='theta 6 Hz')
ax[0].plot(t,gamma_theta*0.25,label='γ (θ‑mod)',alpha=.6)
ax[0].set_xlim(0,0.5); ax[0].set_title('Lexical assembly: θ→γ'); ax[0].legend()

ax[1].plot(t,delta,label='delta 2 Hz',color='orange')
ax[1].plot(t,gamma_delta*0.25,label='γ (δ‑mod)',alpha=.6,color='green')
ax[1].set_xlim(0,0.5); ax[1].set_title('Generic binding: δ→γ'); ax[1].legend()

plt.tight_layout()
plt.show()
