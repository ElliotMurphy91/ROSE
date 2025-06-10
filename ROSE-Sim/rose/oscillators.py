
"""Oscillator primitives for ROSE-Sim.
   We provide:
     * Sinusoid: deterministic sine source
     * WilsonCowanOscillator: simple excitatory–inhibitory population pair
"""
import numpy as np

class Sinusoid:
    def __init__(self, freq_hz, fs=1000):
        self.freq = freq_hz
        self.fs = fs
    def generate(self, dur_s, phase=0.0, amp=1.0):
        t = np.arange(0, dur_s, 1/self.fs)
        return amp * np.sin(2*np.pi*self.freq*t + phase)

class WilsonCowanOscillator:
    """Very light‑weight Wilson–Cowan E‑I model.
    Equations:
        dE/dt = -E + sigma(wEE*E - wEI*I + P)
        dI/dt = -I + sigma(wIE*E - wII*I + Q)
    where sigma(x) = 1/(1 + exp(-a*(x-theta)))
    Integration: Euler.
    """
    def __init__(self, fs=1000, a=1.2, theta=2.0,
                 wEE=9.0, wEI=4.0, wIE=13.0, wII=11.0,
                 P=1.25, Q=0.5):
        self.fs=fs; self.dt=1/fs
        self.params=dict(a=a,theta=theta,wEE=wEE,wEI=wEI,wIE=wIE,wII=wII,P=P,Q=Q)
    def _sig(self,x,a,theta):
        return 1.0/(1+np.exp(-a*(x-theta)))
    def simulate(self, dur_s, E0=0.2, I0=0.0):
        N=int(dur_s*self.fs)
        E=np.zeros(N); I=np.zeros(N)
        E[0]=E0; I[0]=I0
        p=self.params
        for n in range(1,N):
            dE = (-E[n-1] + self._sig(p['wEE']*E[n-1]-p['wEI']*I[n-1]+p['P'],
                                      p['a'],p['theta']))*self.dt
            dI = (-I[n-1] + self._sig(p['wIE']*E[n-1]-p['wII']*I[n-1]+p['Q'],
                                      p['a'],p['theta']))*self.dt
            E[n]=E[n-1]+dE
            I[n]=I[n-1]+dI
        return E,I
