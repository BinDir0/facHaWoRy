"""
Any4D chunked inference wrapper for HaWoR pipeline.

Splits long videos into overlapping chunks, runs Any4D inference per chunk,
and packages outputs for downstream Sim(3) alignment.
"""

import os
import sys
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional
from glob import glob
from natsort import natsorted
import logging

logger = logging.getLogger(__name__)

# Add Any4D to path
ANY4D_ROOT = "/root/Any4D"
if ANY4D_ROOT not in sys.path:
    sys.path.insert(0, ANY4D_ROOT)


def load_any4d_model(config: dict, device: str = "cuda"):
    """
    Load Any4D model and MoGe mask model.

    Args:
        config: dict with keys:
            - path: hydra config path
            - config_overrides: list of overrides
            - checkpoint_path: model checkpoint
            - data_norm_type: image normalization type
            - trained_with_amp: bool
        device: torch device string

    Returns:
        model: Any4D model (eval mode)
        moge_model: MoGe model for mask computation
    """
    from any4d.utils.moge_inference import load_moge_model
    from scripts.demo_inference import init_inference_model

    device = torch.device(device)
    model = init_inference_model(config, config["checkpoint_path"], device)
    moge_model = load_moge_model(device=device)
    return model, moge_model


def _build_chunk_ranges(
    n_frames: int, chunk_size: int, overlap: int
) -> List[Tuple[int, int]]:
    """
    Build (start, end) index ranges for overlapping chunks.

    Returns list of (start_idx, end_idx) tuples where end_idx is exclusive.
    """
    if n_frames <= chunk_size:
        return [(0, n_frames)]

    stride = chunk_size - overlap
    ranges = []
    start = 0
    while start < n_frames:
        end = min(start + chunk_size, n_frames)
        ranges.append((start, end))
        if end == n_frames:
            break
        start += stride
    return ranges


def _extract_chunk_outputs(
    raw_outputs: dict,
    start_idx: int,
    end_idx: int,
) -> Dict:
    """
    Extract per-frame depth, ray_dirs, poses, scene_flow from
    Any4D raw inference outputs and package into chunk dict format.

    The raw_outputs dict has keys "pred1", "pred2", ..., "pred{N+1}"
    where pred1 = reference frame (view 0), pred2..pred{N+1} = sequence frames.

    Input image list was: [ref_path] + chunk_paths  (N+1 views total)
    So: view 0 -> pred1 (reference), view k -> pred{k+1}
    Sequence frames chunk_paths[k] -> view k+1 -> pred{k+2}

    Returns chunk dict with keys:
        depth_along_ray: list of (H, W) np arrays
        ray_dirs: list of (H, W, 3) np arrays
        cam_trans: list of (3,) np arrays
        cam_quats: list of (4,) np arrays (XYZW)
        scene_flow: list of (H, W, 3) np arrays or None
        global_to_local: dict mapping global_frame_idx -> local_idx
    """
    n_frames = end_idx - start_idx
    depths = []
    ray_dirs_list = []
    cam_trans_list = []
    cam_quats_list = []
    scene_flow_list = []
    has_flow = False

    for local_idx in range(n_frames):
        # chunk_paths[local_idx] is view (local_idx + 1) in the input list
        # which maps to pred{local_idx + 2} in the output dict
        pred_key = f"pred{local_idx + 2}"
        pred = raw_outputs[pred_key]

        # Depth along ray: (B, H, W, 1) -> (H, W)
        d = pred["depth_along_ray"][0].cpu().numpy()
        if d.ndim == 3 and d.shape[-1] == 1:
            d = d[..., 0]
        depths.append(d)

        # Ray directions: (B, H, W, 3) -> (H, W, 3)
        rays = pred["ray_directions"][0].cpu().numpy()
        ray_dirs_list.append(rays)

        # Camera translation and quaternion (cam2world)
        # cam_trans: (B, 3) -> (3,), cam_quats: (B, 4) -> (4,) [x, y, z, w]
        cam_t = pred["cam_trans"][0].cpu().numpy()
        cam_q = pred["cam_quats"][0].cpu().numpy()
        cam_trans_list.append(cam_t)
        cam_quats_list.append(cam_q)

        # Scene flow: (B, H, W, 3) if available (not for reference frame)
        if "scene_flow" in pred:
            flow = pred["scene_flow"][0].cpu().numpy()
            scene_flow_list.append(flow)
            has_flow = True
        elif has_flow:
            # Pad with zeros if some views have flow and others don't
            H, W = depths[-1].shape[:2]
            scene_flow_list.append(np.zeros((H, W, 3), dtype=np.float32))

    # Build global_to_local mapping
    global_to_local = {
        global_idx: local_idx
        for local_idx, global_idx in enumerate(range(start_idx, end_idx))
    }

    chunk = {
        "depth_along_ray": depths,
        "ray_dirs": ray_dirs_list,
        "cam_trans": cam_trans_list,
        "cam_quats": cam_quats_list,
        "scene_flow": scene_flow_list if has_flow else None,
        "global_to_local": global_to_local,
    }
    return chunk


def run_chunked_inference(
    image_dir: str,
    model,
    moge_model,
    config: dict,
    chunk_size: int = 180,
    overlap: int = 30,
    device: str = "cuda",
    image_size: int = 518,
) -> Tuple[List[Dict], List[Tuple[int, int]]]:
    """
    Run Any4D inference on a video split into overlapping chunks.

    Args:
        image_dir: directory containing video frames as images
        model: loaded Any4D model
        moge_model: loaded MoGe model
        config: Any4D config dict (needs data_norm_type, trained_with_amp)
        chunk_size: max frames per chunk
        overlap: overlap between consecutive chunks
        device: torch device
        image_size: image resize dimension for model input

    Returns:
        chunks: List of chunk dicts, each containing:
            depth_along_ray, ray_dirs, cam_trans, cam_quats,
            scene_flow, global_to_local
        chunk_ranges: List of (start_frame, end_frame) tuples
    """
    from any4d.utils.image import load_images
    from scripts.demo_inference import sample_inference

    # Gather and sort image paths
    exts = ("*.png", "*.jpg", "*.jpeg")
    image_paths = []
    for ext in exts:
        image_paths.extend(glob(os.path.join(image_dir, ext)))
    image_paths = natsorted(image_paths)
    n_frames = len(image_paths)
    logger.info(f"Found {n_frames} frames in {image_dir}")

    if n_frames == 0:
        raise ValueError(f"No images found in {image_dir}")

    # Build chunk ranges
    chunk_ranges = _build_chunk_ranges(n_frames, chunk_size, overlap)
    logger.info(
        f"Split into {len(chunk_ranges)} chunks "
        f"(size={chunk_size}, overlap={overlap})"
    )

    device_obj = torch.device(device)
    use_amp = config.get("trained_with_amp", True)
    chunks = []

    for chunk_idx, (start, end) in enumerate(chunk_ranges):
        logger.info(
            f"Processing chunk {chunk_idx+1}/{len(chunk_ranges)}: "
            f"frames [{start}, {end})"
        )

        # Build image list: first image is reference frame (view 0),
        # then all chunk frames as sequence (views 1..N).
        # Reference = first frame of chunk; it also appears in sequence.
        chunk_paths = image_paths[start:end]
        ref_path = chunk_paths[0]
        image_list = [ref_path] + chunk_paths

        # Load images with MoGe mask computation
        # API: load_images(folder_or_list, norm_type=, resolution_set=, ...)
        input_views = load_images(
            image_list,
            norm_type=config.get("data_norm_type", "dinov2"),
            resolution_set=image_size,
            verbose=False,
            compute_moge_mask=True,
            moge_model=moge_model,
        )

        # Move to device
        for view in input_views:
            for k, v in view.items():
                if isinstance(v, torch.Tensor):
                    view[k] = v.to(device_obj)

        # Run inference
        # API: sample_inference(model, views, device, use_amp)
        # It handles AMP internally via use_amp parameter (bf16 by default)
        with torch.no_grad():
            raw_outputs = sample_inference(
                model, input_views, device_obj, use_amp
            )

        # Extract chunk dict directly from raw outputs
        # raw_outputs has keys: "pred1", "pred2", ..., "view1", "view2", ..., "loss"
        chunk = _extract_chunk_outputs(raw_outputs, start, end)
        chunks.append(chunk)

        # Free GPU memory
        del raw_outputs, input_views
        torch.cuda.empty_cache()

    return chunks, chunk_ranges
