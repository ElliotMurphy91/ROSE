<p align="center">
  <b>A lightweight sandbox for <em>ROSE</em> – the <br>
  <span style="font-size:1.3em;color:#e63946;"><b>R</b></span>epresentation
  <span style="font-size:1.3em;color:#f4a261;"><b>O</b></span>peration
  <span style="font-size:1.3em;color:#2a9d8f;"><b>S</b></span>tructure
  <span style="font-size:1.3em;color:#457b9d;"><b>E</b></span>ncoding architecture for syntax</b>
</p>

---

<h1 align="center"><img src="https://elliotmurphyblog.wordpress.com/wp-content/uploads/2025/06/figure6.jpg" alt="ROSE-Sim"></h1>


---

## 🚀 Quick Start
# 1. Install deps
pip install -r requirements.txt

# 2. Run a minimal MERGE demo
python experiments/two_word_merge.py

## 🌐 Repo layout
```text
ROSE-Sim/
├─ rose/                    core package
│  ├─ __init__.py
│  ├─ oscillators.py        # Sinusoid + Wilson-Cowan E–I node
│  ├─ workspace.py          # δ–θ push/reset & α-load logic
│  └─ analysis.py           # MI, PLV, PAC helpers
├─ experiments/             runnable demos
│  ├─ two_word_merge.py     # headedness + workspace
│  ├─ operation_level_coupling.py  # lexical feature bundling
│  └─ head_competition_travelwave.py  # θ–γ competition & δ wave
│  └─ left_corner_parser_pac.py		# modeling LC parser via ROSE
│  └─ motif_binding_demo.py	# demo of dynamical motifs emerging from spike-phase coupling
│  └─ traveling_wave_workspace.py	# model for basics of the E component of ROSE  
├─ tests/
│  └─ test_pac.py
├─ requirements.txt
└─ README.md
