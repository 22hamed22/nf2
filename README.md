# NeRF Example (refactored)

This repository contains a refactored Jupyter example to train a small NeRF model.

Quick status
- src/ contains modularized code:
  - src/data_loader.py — dataset loading and ray helpers
  - src/model.py — positional encoding + model classes
  - src/train.py — forward pass and training helpers
  - src/utils.py — rendering & sampling helpers
  - src/__init__.py — convenient re-exports
- notebooks/ contains the original notebook. Consider running the refactored notebook from the repo root so Python finds the src package.
- docs/ contains basic documentation pages.

Requirements
- Python 3.9+
- CUDA-enabled PyTorch (if you want to train on GPU)
- See requirements.txt for a recommended pip install list and notes.

Getting started (example)
1. Create and activate a virtual environment:
   python -m venv .venv
   source .venv/bin/activate   # (Linux/macOS)
   .venv\Scripts\Activate.ps1  # (Windows PowerShell)

2. Install pip packages:
   pip install -r requirements.txt

3. Install torchsearchsorted (the notebook also installs it inline):
   pip install git+https://github.com/aliutkus/torchsearchsorted

4. From the repository root (E:\nerf2), run Jupyter or open the notebook:
   jupyter notebook notebooks/full_nerf_example.ipynb
   or run the refactored notebook if you saved it as notebooks/full_nerf_example_refactored.ipynb

Notes
- Run notebooks from the repo root so imports like `from src.model import *` work.
- The training loop in the notebook is long (hours) and expects GPU. For quick debugging reduce iterations and image size (downscale factor).
- If you want the code to be packaging-friendly, I can convert src imports to relative imports (from .utils import ...) and add a setup.py/pyproject.toml.

License
- Add a LICENSE file if you plan to publish. Currently no license file is included.

If you'd like, I can:
- convert imports in modules to relative imports,
- move the notebook training loop into a train() function in src/train.py,
- generate a saved refactored notebook file you can download.