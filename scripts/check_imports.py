#!/usr/bin/env python3
"""
Quick smoke test: import refactored modules and run a single forward pass.
Run from repository root (E:\nerf2):
  python .\scripts\check_imports.py
"""
import sys
import torch
import numpy as np
from pathlib import Path

# Ensure repo root is on path (optional if you run from repo root)
repo_root = Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

try:
    import src  # tests package __init__
    from src import load_blender_data, get_ray_bundle, VeryTinyNeRFModel, positional_encoding, nerf_forward_pass, img2mse, mse2psnr
    print("Imported src package and definitions from src.")
except Exception as e:
    print("ERROR importing src package or modules:", e)
    raise

device = "cpu"  # Use CPU for smoke test (works even without CUDA)

# tiny fake dataset configuration
H, W = 16, 16
focal = 1.0

# create a fake camera pose (identity 4x4)
pose = torch.eye(4, dtype=torch.float32)

# Build rays for tiny image
ray_origins, ray_directions = get_ray_bundle(H, W, focal, pose)
batch_rays = torch.stack([ray_origins, ray_directions], dim=0)  # shape (2, W, H, 3)

# instantiate model (tiny) — make encoding sizes match the encoder used below
model_coarse = VeryTinyNeRFModel(num_encoding_fn_xyz=1, num_encoding_fn_dir=1)
model_coarse.to(device)

# positional encoders (minimal, call underlying function directly)
encode_position_fn = lambda x: positional_encoding(x, num_encoding_functions=1, include_input=True)
encode_direction_fn = lambda x: positional_encoding(x, num_encoding_functions=1, include_input=True)

# run single forward pass (num_fine=0 to avoid hierarchical sampling)
try:
    rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine = nerf_forward_pass(
        H=H, W=W, focal=focal,
        model_coarse=model_coarse, model_fine=None,
        batch_rays=batch_rays,
        use_viewdirs=True,
        near=0.0, far=1.0,
        chunksize=1024, num_coarse=8, num_fine=0,
        mode="validation", lindisp=False, perturb=False,
        encode_position_fn=encode_position_fn,
        encode_direction_fn=encode_direction_fn,
        radiance_field_noise_std=0., white_background=False
    )
    print("Forward pass succeeded.")
    print("rgb_coarse shape:", getattr(rgb_coarse, 'shape', None))
    if rgb_fine is None:
        print("rgb_fine is None (expected for num_fine=0).")
except Exception as e:
    print("ERROR during forward pass:", e)
    raise

print("Smoke test done.")