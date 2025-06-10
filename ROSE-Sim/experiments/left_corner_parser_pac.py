"""
LEFT-CORNER PARSING × δ–θ PHASE
-----------------------------------------------------
Action-to-phase mapping (hard-coded):

  δ trough      → SHIFT   (lexical retrieval)    blue
  θ local peak  → NT      (open phrase)          orange
  δ peak        → REDUCE  (Merge + Label)        crimson + β-burst

Words are spaced 400 ms apart for clarity.
"""
import numpy as np, matplotlib.pyplot as plt
from rose.workspace import SyntaxWorkspace

# ---------------- scaffold ----------------
fs, dur = 1000, 3            # Hz, seconds
t  = np.arange(0, dur, 1/fs)
delta = np.sin(2*np.pi*2*t)                    # 2 Hz
theta = np.sin(2*np.pi*6*t + np.pi/4)          # 6 Hz, 45° offset

# helper: find nearest sample to a phase condition
def nearest(samples, centre):
    return samples[np.argmin(np.abs(samples - centre))]

# compute phase landmarks (samples):
δ_troughs = np.where((np.r_[False, np.diff(np.signbit(delta))] > 0))[0]
δ_peaks   = np.where((np.r_[False, np.diff(np.signbit(delta))] < 0))[0]
θ_peaks   = np.where((np.r_[False, np.diff(np.signbit(np.diff(theta)))] < 0))[0]

# ------------------------------------------
actions = ['SHIFT','NT','SHIFT','REDUCE','REDUCE']    # 5-word toy sentence
word_onsets = np.arange(0.2, 2.2, 0.4)                # 200-2000 ms

phase_targets = {
    'SHIFT'  : δ_troughs,
    'NT'     : θ_peaks,
    'REDUCE' : δ_peaks
}
col = {'SHIFT':'dodgerblue','NT':'orange','REDUCE':'crimson'}

ws = SyntaxWorkspace(capacity=6)
beta_idx = []

print("w(ms) act   stack  Δt(ms)")
for w, act in zip(word_onsets, actions):
    idx_onset = int(w*fs)
    tgt_idx   = nearest(phase_targets[act], idx_onset)
    lag_ms    = (idx_onset - tgt_idx)/fs*1000

    # warn if mis-aligned >40 ms
    if abs(lag_ms) > 40:
        print(f" WARNING: {act} mis-aligned by {lag_ms:+.0f} ms")

    # enforce perfect alignment for plotting
    idx = tgt_idx

    if act == 'SHIFT':
        ws.push('LEX')
    elif act == 'NT':
        ws.push('(')
    elif act == 'REDUCE':
        ws.pop(); beta_idx.append(idx)

    print(f"{t[idx]*1000:4.0f} {act:6}  {ws.depth():3}   {lag_ms:+.0f}")

    plt.axvline(t[idx], color=col[act], ls='--', alpha=.8)

# ---------------- plot scaffold & bursts ----------------
plt.plot(t, delta, label='δ (2 Hz)', lw=1)
plt.plot(t, theta, label='θ (6 Hz)', lw=1)

for b in beta_idx:
    plt.plot(t[b], 0, 'k*', ms=9, zorder=10)

plt.title("Phase-coded left-corner parsing")
plt.xlabel('time (s)'); plt.ylim(-1.2, 1.2); plt.xlim(0, 2.4)
plt.legend(); plt.tight_layout(); plt.show()
