Simulation of some core components of the ROSE neurocomputational architecture for syntax.

ROSE-Sim/
├─ rose/                    (core package)
│  ├─ __init__.py
│  ├─ oscillators.py        # Sinusoid + Wilson-Cowan E–I node
│  ├─ workspace.py          # push / phase-reset / α-load logic
│  └─ analysis.py           # modulation-index, PLV
├─ experiments/
│  └─ two_word_merge.py     # headedness + workspace demo
│  └─ operation_level_coupling.py	# lexical feature-bundling and generic combinatorics
│  └─ head_competition.py	# theta-gamma MI increases determining headedness
│  └─ left_corner_parser_pac.py		# modeling LC parser via ROSE
│  └─ motif_binding_demo.py	# demo of dynamical motifs emerging from spike-phase coupling
│  └─ traveling_wave_workspace.py	# model for basics of the E component of ROSE  
├─ tests/
│  └─ test_pac.py
├─ requirements.txt

Examples:
How to run demo:
pip install -r requirements.txt
python experiments/two_word_merge.py

What you should see:
MI1=0.239 MI2=0.272 → head=FB2
Workspace depth: 2 alpha load= 0.5

For operation_level_coupling demo:
python experiments/operation_level_coupling.py

Then tweak:
# Increase lexical PAC
theta_boost = 1.8          # in script, multiply gamma_theta amplitude
# Add noise
theta += 0.2*np.random.randn(len(theta))
