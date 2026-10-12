# Optimal Gating for Dead-Time–Constrained Single-Photon Detectors

This repository contains the code to reproduce the results of *Optimal Gating for Dead-Time–Constrained Single-Photon Detectors*.

Install the Python dependencies and a LaTeX distribution with `dvipng`, `amsmath`, `amssymb`, `type1cm`, and `type1ec`:

```sh
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Compute and plot the results:

```sh
python regret.py --workers 4     # Figure 2
python suffix.py --workers 12    # Figure 4, including the 64 × 64 heatmap
python remarks.py               # Relative error and optical attenuation
python mc.py                    # Monte Carlo validation
```

The scripts save results and figures in `output/`. Use `--plot` with `regret.py`, `suffix.py`, or `mc.py` to redraw existing results without recomputing them. Use `--heatmap 4` with `suffix.py` for a smaller trial grid.
