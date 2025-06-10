
"""Two-word Merge/headedness demo using ROSE-Sim primitives."""
import numpy as np
import matplotlib.pyplot as plt
from rose.oscillators import Sinusoid
from rose.analysis import modulation_index
from rose.workspace import SyntaxWorkspace

fs=1000; dur=3.0
t=np.arange(0,dur,1/fs)
delta=Sinusoid(2,fs).generate(dur)
theta1=Sinusoid(6,fs).generate(dur)
theta2=Sinusoid(6,fs).generate(dur,amp=1.5)

gamma1=np.maximum(theta1,0)*Sinusoid(80,fs).generate(dur)
gamma2=np.maximum(theta2,0)*Sinusoid(80,fs).generate(dur)

mi1=modulation_index(delta,gamma1)
mi2=modulation_index(delta,gamma2)
head='FB1' if mi1>mi2 else 'FB2'
print(f'MI1={mi1:.3f} MI2={mi2:.3f} → head={head}')

# workspace demonstration
ws=SyntaxWorkspace(capacity=4)
ws.push('NP')
ws.push('VP')
print('Workspace depth:',ws.depth(),'alpha load=',ws.alpha_load())

plt.figure(figsize=(8,4))
plt.plot(t,delta,label='δ')
plt.plot(t,theta1,label='θ1')
plt.plot(t,theta2,label='θ2')
plt.xlim(0,0.5)
plt.legend(); plt.tight_layout(); plt.show()
