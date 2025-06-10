<h1 align="center"><img src="https://img.shields.io/badge/ROSE-Sim-%F0%9F%8C%B9%20Recursive%20Oscillatory%20Syntax%20Engine-5e9?style=for-the-badge&logoColor=white" alt="ROSE-Sim"></h1>

<p align="center">
  <b>A lightweight sandbox for <em>ROSE</em> 🌹 – the <br>
  <span style="font-size:1.3em;color:#e63946;"><b>R</b></span>epresentation-
  <span style="font-size:1.3em;color:#f4a261;"><b>O</b></span>peration-
  <span style="font-size:1.3em;color:#2a9d8f;"><b>S</b></span>tructure-
  <span style="font-size:1.3em;color:#457b9d;"><b>E</b></span>ncoding architecture for syntax</b>
</p>

---

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
├─ tests/
│  └─ test_pac.py
├─ requirements.txt
└─ README.md
