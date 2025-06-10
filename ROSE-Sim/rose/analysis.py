
"""Analysis helpers: phase–amplitude coupling, PLV, etc."""
import numpy as np
from scipy.signal import hilbert

__all__=['modulation_index','phase_locking_value']

def modulation_index(low_sig, high_sig, n_bins=18):
    """Tort *et al.* MI."""
    phase=np.angle(hilbert(low_sig))
    amp=np.abs(hilbert(high_sig))
    bins=np.linspace(-np.pi,np.pi,n_bins+1)
    mean_amp=np.array([amp[(phase>=bins[i])&(phase<bins[i+1])].mean()
                       for i in range(n_bins)])
    mean_amp=np.nan_to_num(mean_amp)
    p=mean_amp/mean_amp.sum()
    p=np.where(p==0,1e-12,p)
    mi=(np.log(n_bins)+np.sum(p*np.log(p)))/np.log(n_bins)
    return mi

def phase_locking_value(sig1, sig2):
    phi1=np.angle(hilbert(sig1))
    phi2=np.angle(hilbert(sig2))
    return np.abs(np.exp(1j*(phi1-phi2)).mean())
