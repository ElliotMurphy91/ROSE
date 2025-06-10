# experiments/head_competition_travelwave.py
"""
Two competing theta–gamma generators nested within a common delta pacemaker.

• Each θ–γ module (A or B) is a phase–amplitude–coupled oscillator.
• PAC strength is tunable; strongest coupling ⇒ HEAD.
• The winning module sets the spatial phase-gradient, i.e. the direction
  (angle) of a propagating δ travelling wave, echoing feature projection.

This demo illustrates the 'push-reset-push' cycle and links it to syntactic headedness.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert

# ---------------------------------------------------------------------
# Parameters
fs          = 1000          # Hz
T           = 2.0           # seconds
t           = np.arange(0, T, 1/fs)

f_delta     = 2             # Hz  (δ pacemaker)
f_theta     = 6             # Hz  (θ carriers)
f_gamma     = 70            # Hz  (γ bursts)

# Relative PAC strengths  (0–1); tweak to flip dominance
pac_A       = 0.8           # θA↔δ coupling weight
pac_B       = 0.3           # θB↔δ coupling weight

noise       = 0.15

# ---------------------------------------------------------------------
# Oscillator definitions
delta = np.sin(2*np.pi*f_delta*t)

def theta_gamma(pac_strength, phase_shift=0):
    """Generate a θ carrier whose γ amplitude is modulated by δ phase."""
    theta = np.sin(2*np.pi*f_theta*t + phase_shift)
    # γ bursts follow θ peaks; amplitude further scaled by δ * pac_strength
    gamma_amp = (1 + pac_strength*delta)/2          # 0–1 range
    gamma = gamma_amp * np.sin(2*np.pi*f_gamma*t + phase_shift)
    return theta + gamma + noise*np.random.randn(len(t))

theta_A = theta_gamma(pac_A, phase_shift=0)
theta_B = theta_gamma(pac_B, phase_shift=np.pi/3)   # spatial offset

# ---------------------------------------------------------------------
# Phase-amplitude coupling (MI) per generator
def pac_mi(low, high):
    pha = np.angle(hilbert(low))
    amp = np.abs(hilbert(high))
    nbin = 18
    bins = np.linspace(-np.pi, np.pi, nbin+1)
    amp_binned = np.array([amp[(pha>=bins[i])&(pha<bins[i+1])].mean()
                           for i in range(nbin)])
    p = amp_binned/amp_binned.sum()
    h = -np.nansum(p*np.log(p+1e-12))
    h_max = np.log(nbin)
    return (h_max-h)/h_max    # modulation index (0–1)

mi_A = pac_mi(delta, theta_A)
mi_B = pac_mi(delta, theta_B)

head = "θ_A" if mi_A > mi_B else "θ_B"
dominant_mi = max(mi_A, mi_B)

# Map dominance to travelling-wave angle (rad)
# simple heuristic: larger MI ⇒ earlier phase lead ⇒ forward wave (+y)
angle = np.interp(dominant_mi, [0,1], [-np.pi/4, np.pi/4])   # −45°..+45°

# ---------------------------------------------------------------------
# Plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 6), sharex=True)

axes[0].plot(t, delta, 'k', label='δ 2 Hz')
axes[0].set_ylabel('δ')
axes[0].legend(loc='upper right')

axes[1].plot(t, theta_A, 'C1', alpha=0.8, label=f'θ_A (PAC={pac_A})')
axes[1].plot(t, theta_B, 'C2', alpha=0.8, label=f'θ_B (PAC={pac_B})')
axes[1].set_ylabel('θ + γ')
axes[1].legend(loc='upper right')

axes[2].bar(['θ_A','θ_B'], [mi_A, mi_B], color=['C1','C2'])
axes[2].set_ylabel('δ–θ MI')
axes[2].set_title(f'Head = {head} (angle={np.degrees(angle):.1f}°)')

plt.tight_layout()
plt.show()

# ---------------------------------------------------------------------
# Console summary
print("δ–θ PAC (MI):  A = %.3f,  B = %.3f" % (mi_A, mi_B))
print(f"Winner (head): {head}")
print("Travelling-wave angle (deg): %.1f" % np.degrees(angle))
