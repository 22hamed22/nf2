# Auto-extracted training / forward orchestration for NeRF
import torch
import math
from typing import Callable, Optional

from src.utils import volume_render_radiance_field, sample_pdf, ndc_rays, meshgrid_xy


def get_minibatches(inputs: torch.Tensor, chunksize: int = 1024 * 8):
    """Split a huge tensor into list of minibatches along dim 0."""
    return [inputs[i:i + chunksize] for i in range(0, inputs.shape[0], chunksize)]


def run_network(network_fn: Callable, pts, ray_batch, chunksize, embed_fn: Callable,
                embeddirs_fn: Optional[Callable]):
    pts_flat = pts.reshape((-1, pts.shape[-1]))
    embedded = embed_fn(pts_flat)
    if embeddirs_fn is not None:
        viewdirs = ray_batch[..., None, -3:]
        input_dirs = viewdirs.expand(pts.shape)
        input_dirs_flat = input_dirs.reshape((-1, input_dirs.shape[-1]))
        embedded_dirs = embeddirs_fn(input_dirs_flat)
        embedded = torch.cat((embedded, embedded_dirs), dim=-1)

    batches = get_minibatches(embedded, chunksize=chunksize)
    preds = []
    for batch in batches:
        preds.append(network_fn(batch))
    radiance_field = torch.cat(preds, dim=0)
    radiance_field = radiance_field.reshape(list(pts.shape[:-1]) + [radiance_field.shape[-1]])
    return radiance_field


def predict_and_render_radiance(
    ray_batch, model_coarse, model_fine, num_coarse, num_fine, chunksize,
    mode="train", lindisp=False, perturb=True,
    encode_position_fn=None,
    encode_direction_fn=None,
    radiance_field_noise_std=0.,
    white_background=False
):
    num_rays = ray_batch.shape[0]
    ro, rd = ray_batch[..., :3], ray_batch[..., 3:6]
    bounds = ray_batch[..., 6:8].reshape((-1, 1, 2))
    near, far = bounds[..., 0], bounds[..., 1]

    t_vals = torch.linspace(0., 1., num_coarse).to(ro)
    if not lindisp:
        z_vals = near * (1. - t_vals) + far * t_vals
    else:
        z_vals = 1. / (1. / near * (1. - t_vals) + 1. / far * t_vals)
    z_vals = z_vals.expand([num_rays, num_coarse])

    if perturb:
        mids = 0.5 * (z_vals[..., 1:] + z_vals[..., :-1])
        upper = torch.cat((mids, z_vals[..., -1:]), dim=-1)
        lower = torch.cat((z_vals[..., :1], mids), dim=-1)
        t_rand = torch.rand(z_vals.shape).to(ro)
        z_vals = lower + (upper - lower) * t_rand

    pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

    radiance_field = run_network(model_coarse,
                                 pts,
                                 ray_batch,
                                 chunksize,
                                 encode_position_fn,
                                 encode_direction_fn,
                                )

    rgb_coarse, disp_coarse, acc_coarse, weights, depth_coarse = volume_render_radiance_field(
        radiance_field, z_vals, rd,
        radiance_field_noise_std=radiance_field_noise_std,
        white_background=white_background
    )

    rgb_fine, disp_fine, acc_fine = None, None, None
    if num_fine > 0:
        z_vals_mid = .5 * (z_vals[..., 1:] + z_vals[..., :-1])
        z_samples = sample_pdf(
            z_vals_mid, weights[..., 1:-1], num_fine,
            det=(perturb == 0.)
        )
        z_samples = z_samples.detach()

        z_vals, _ = torch.sort(torch.cat((z_vals, z_samples), dim=-1), dim=-1)
        pts = ro[..., None, :] + rd[..., None, :] * z_vals[..., :, None]

        radiance_field = run_network(model_fine,
                                     pts,
                                     ray_batch,
                                     chunksize,
                                     encode_position_fn,
                                     encode_direction_fn
                                    )
        rgb_fine, disp_fine, acc_fine, _, _ = volume_render_radiance_field(
            radiance_field, z_vals, rd,
            radiance_field_noise_std=radiance_field_noise_std,
            white_background=white_background
        )

    return rgb_coarse, disp_coarse, acc_coarse, rgb_fine, disp_fine, acc_fine


def nerf_forward_pass(
    H, W, focal, model_coarse, model_fine, batch_rays,
    use_viewdirs, near, far, chunksize, num_coarse, num_fine, mode="train",
    lindisp=False, perturb=True, encode_position_fn=None,
    encode_direction_fn=None, radiance_field_noise_std=0.,
    white_background=False
):
    ray_origins = batch_rays[0]
    ray_directions = batch_rays[1]
    if use_viewdirs:
        viewdirs = ray_directions
        viewdirs = viewdirs / viewdirs.norm(p=2, dim=-1).unsqueeze(-1)
        viewdirs = viewdirs.reshape((-1, 3))
    ray_shapes = ray_directions.shape
    ro, rd = ndc_rays(H, W, focal, 1., ray_origins, ray_directions)
    ro = ro.reshape((-1, 3))
    rd = rd.reshape((-1, 3))
    near = near * torch.ones_like(rd[..., :1])
    far = far * torch.ones_like(rd[..., :1])
    rays = torch.cat((ro, rd, near, far), dim=-1)
    if use_viewdirs:
        rays = torch.cat((rays, viewdirs), dim=-1)

    batches = get_minibatches(rays, chunksize=chunksize)
    rgb_coarse, disp_coarse, acc_coarse = [], [], []
    rgb_fine, disp_fine, acc_fine = None, None, None
    for batch in batches:
        rc, dc, ac, rf, df, af = predict_and_render_radiance(
            batch, model_coarse, model_fine, num_coarse, num_fine, chunksize,
            mode, lindisp=lindisp, perturb=perturb,
            encode_position_fn=encode_position_fn,
            encode_direction_fn=encode_direction_fn,
            radiance_field_noise_std=radiance_field_noise_std,
            white_background=white_background
        )
        rgb_coarse.append(rc)
        disp_coarse.append(dc)
        acc_coarse.append(ac)
        if rf is not None:
            if rgb_fine is None:
                rgb_fine = [rf]
            else:
                rgb_fine.append(rf)
        if df is not None:
            if disp_fine is None:
                disp_fine = [df]
            else:
                disp_fine.append(df)
        if af is not None:
            if acc_fine is None:
                acc_fine = [af]
            else:
                acc_fine.append(af)

    rgb_coarse_ = torch.cat(rgb_coarse, dim=0)
    disp_coarse_ = torch.cat(disp_coarse, dim=0)
    acc_coarse_ = torch.cat(acc_coarse, dim=0)
    if rgb_fine is not None:
        rgb_fine_ = torch.cat(rgb_fine, dim=0)
    else:
        rgb_fine_ = None
    if disp_fine is not None:
        disp_fine_ = torch.cat(disp_fine, dim=0)
    else:
        disp_fine_ = None
    if acc_fine is not None:
        acc_fine_ = torch.cat(acc_fine, dim=0)
    else:
        acc_fine_ = None

    return rgb_coarse_, disp_coarse_, acc_coarse_, rgb_fine_, disp_fine_, acc_fine_