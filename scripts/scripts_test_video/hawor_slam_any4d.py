"""
Any4D-based SLAM replacement for HaWoR pipeline.

Replaces DROID-SLAM + Metric3D with Any4D feed-forward inference +
Sim(3) chunk alignment + optional Pose Graph Optimization.

Produces the same NPZ output format as hawor_slam.py for downstream
compatibility with the infiller stage.
"""

import os
import sys
import time
import logging
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(__file__) + '/../..')

from lib.pipeline.frame_source import build_frame_source

logger = logging.getLogger(__name__)

# Check if we should suppress verbose output
QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"


def vprint(*args, **kwargs):
    """Print only if not in quiet mode."""
    if not QUIET_MODE:
        print(*args, **kwargs)


# ---------------------------------------------------------------------------
# Default Any4D config (can be overridden by the caller)
# ---------------------------------------------------------------------------

DEFAULT_ANY4D_CONFIG = {
    "path": "configs/any4d_video",
    "config_overrides": [],
    "checkpoint_path": "/root/Any4D/checkpoints/any4d.ckpt",
    "data_norm_type": "dinov2",
    "trained_with_amp": True,
}


# ---------------------------------------------------------------------------
# Helper: fuse aligned chunk data into per-frame arrays
# ---------------------------------------------------------------------------

def _fuse_chunk_data(chunks, chunk_ranges):
    """
    Fuse per-frame data from aligned chunks using hard_switch at overlap midpoints.

    For non-overlapping frames, uses the only available data.
    For overlapping frames, uses data from the chunk whose center is closer
    (equivalently: hard switch at the midpoint of the overlap region).

    Returns:
        all_trans: list of (3,) translations, one per frame
        all_quats: list of (4,) quaternions [x,y,z,w], one per frame
        all_depths: list of (H, W) depth arrays, one per frame
        all_rays: list of (H, W, 3) ray direction arrays, one per frame
        all_flows: list of (H, W, 3) scene flow arrays or None
    """
    n_total = chunk_ranges[-1][1]

    # For each frame, determine which chunk to use
    frame_source_chunk = {}  # global_idx -> chunk_index
    for i, (start, end) in enumerate(chunk_ranges):
        for global_idx in range(start, end):
            if global_idx not in frame_source_chunk:
                frame_source_chunk[global_idx] = i
            else:
                # Overlap region: switch at midpoint
                prev_start, prev_end = chunk_ranges[i - 1]
                overlap_mid = (start + prev_end) // 2
                if global_idx >= overlap_mid:
                    frame_source_chunk[global_idx] = i

    # Extract per-frame data
    all_trans = []
    all_quats = []
    all_depths = []
    all_rays = []
    all_flows = []
    has_flow = False

    for global_idx in range(n_total):
        chunk_idx = frame_source_chunk[global_idx]
        chunk = chunks[chunk_idx]
        local_idx = chunk["global_to_local"][global_idx]

        all_trans.append(chunk["cam_trans"][local_idx])
        all_quats.append(chunk["cam_quats"][local_idx])

        depth = chunk["depth_along_ray"][local_idx]
        if depth.ndim == 3:
            depth = depth[..., 0]
        all_depths.append(depth)

        all_rays.append(chunk["ray_dirs"][local_idx])

        if chunk.get("scene_flow") is not None:
            all_flows.append(chunk["scene_flow"][local_idx])
            has_flow = True

    return all_trans, all_quats, all_depths, all_rays, (all_flows if has_flow else None)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def hawor_slam_any4d(
    args,
    start_idx,
    end_idx,
    any4d_config=None,
    chunk_size=180,
    overlap=30,
    image_size=518,
    run_pgo=False,
    device="cuda",
):
    """
    Any4D-based SLAM replacement for HaWoR pipeline.

    Runs Any4D chunked inference → Sim(3) alignment → (optional PGO) →
    produces NPZ output compatible with the downstream infiller stage.

    Args:
        args: argparse namespace with video_path, img_focal
        start_idx, end_idx: track range indices (from detect_track)
        any4d_config: dict with Any4D model config (see DEFAULT_ANY4D_CONFIG)
        chunk_size: max frames per Any4D inference chunk
        overlap: overlap between consecutive chunks (for Sim(3) alignment)
        image_size: image resize dimension for Any4D input
        run_pgo: whether to run Sim(3) Pose Graph Optimization
        device: torch device string

    Returns:
        save_path: path to the saved NPZ file
    """
    timing = {}
    t_start = time.time()

    # Resolve config
    config = dict(DEFAULT_ANY4D_CONFIG)
    if any4d_config is not None:
        config.update(any4d_config)

    # File and folders
    file = args.video_path
    video_root = os.path.dirname(file)
    video = os.path.basename(file).split('.')[0]
    seq_folder = os.path.join(video_root, video)
    os.makedirs(seq_folder, exist_ok=True)

    # Frame source (pre-extracted images)
    frame_source = build_frame_source(file)
    n_frames = len(frame_source)
    first_img = frame_source.get_frame(0, rgb=False)
    height, width, _ = first_img.shape

    vprint(f'[Any4D-SLAM] Running on {seq_folder} ({n_frames} frames)')

    # Estimate calibration for NPZ output (not used by Any4D itself)
    focal = args.img_focal
    if focal is None:
        try:
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'r') as f:
                focal = float(f.read())
        except Exception:
            vprint('[Any4D-SLAM] No focal length provided, using max(h,w) as estimate')
            focal = float(max(height, width))
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'w') as f:
                f.write(str(focal))
    center = np.array([width / 2.0, height / 2.0])

    # ---- Step 1: Load Any4D model ----
    t0 = time.time()
    from lib.pipeline.any4d_inference import load_any4d_model, run_chunked_inference
    model, moge_model = load_any4d_model(config, device=device)
    timing['1_load_model'] = time.time() - t0
    vprint(f'[Any4D-SLAM] Model loaded in {timing["1_load_model"]:.1f}s')

    # ---- Step 2: Chunked inference ----
    t0 = time.time()
    # Use the pre-extracted images directory directly
    extracted_dir = os.path.join(seq_folder, "extracted_images")
    if not os.path.isdir(extracted_dir):
        raise FileNotFoundError(
            f"Pre-extracted frames not found at: {extracted_dir}\n"
            f"Run frame extraction first."
        )

    chunks, chunk_ranges = run_chunked_inference(
        extracted_dir, model, moge_model, config,
        chunk_size=chunk_size,
        overlap=overlap,
        device=device,
        image_size=image_size,
    )
    timing['2_inference'] = time.time() - t0
    vprint(
        f'[Any4D-SLAM] Inference done in {timing["2_inference"]:.1f}s '
        f'({len(chunks)} chunks)'
    )

    # Free model GPU memory
    del model, moge_model
    torch.cuda.empty_cache()

    # ---- Step 3: Sim(3) chunk alignment ----
    t0 = time.time()
    if len(chunks) > 1:
        from lib.pipeline.sim3_alignment import align_chunk_pair

        for i in range(1, len(chunks)):
            prev_start, prev_end = chunk_ranges[i - 1]
            curr_start, curr_end = chunk_ranges[i]
            overlap_indices = list(range(curr_start, prev_end))

            if len(overlap_indices) == 0:
                raise RuntimeError(
                    f"No overlap between chunk {i-1} [{prev_start},{prev_end}) "
                    f"and chunk {i} [{curr_start},{curr_end})"
                )

            R, t, s, inlier_ratio = align_chunk_pair(
                chunks[i - 1], chunks[i], overlap_indices,
            )
            vprint(
                f'[Any4D-SLAM] Chunk {i-1}→{i}: '
                f's={s:.4f}, inlier={inlier_ratio:.2%}, '
                f'overlap={len(overlap_indices)} frames'
            )
    else:
        vprint('[Any4D-SLAM] Single chunk, no alignment needed')

    timing['3_alignment'] = time.time() - t0

    # ---- Step 4: Fuse aligned chunks ----
    t0 = time.time()
    all_trans, all_quats, all_depths, all_rays, all_flows = \
        _fuse_chunk_data(chunks, chunk_ranges)

    # Verify frame count
    assert len(all_trans) == n_frames, \
        f"Fused {len(all_trans)} frames but video has {n_frames}"

    # Verify ray directions are unit-normalized (required for depth conversion)
    sample_rays = all_rays[0]
    ray_norms = np.linalg.norm(sample_rays, axis=-1)
    if not np.allclose(ray_norms, 1.0, atol=1e-3):
        logger.warning(
            f"Ray direction norms range [{ray_norms.min():.4f}, {ray_norms.max():.4f}], "
            f"expected ~1.0. depth_along_ray -> z_depth conversion may be inaccurate."
        )

    timing['4_fusion'] = time.time() - t0

    # ---- Step 5: Optional Pose Graph Optimization ----
    t0 = time.time()
    if run_pgo and n_frames > 1:
        from lib.pipeline.pose_graph_optimizer import build_and_run_pgo
        from lib.pipeline.loop_closure_detector import compute_loop_closure_edges

        vprint('[Any4D-SLAM] Running loop closure detection...')
        # Build depth map dict for loop closure
        depth_maps = {i: all_depths[i] for i in range(n_frames)}

        loop_edges = compute_loop_closure_edges(
            all_trans, all_quats,
            depth_maps, all_rays,
            scene_flow_list=all_flows,
        )
        vprint(f'[Any4D-SLAM] Found {len(loop_edges)} loop closures')

        vprint('[Any4D-SLAM] Running Sim(3) PGO...')
        opt_trans, opt_quats, opt_scales = build_and_run_pgo(
            all_trans, all_quats,
            loop_closure_edges=loop_edges if loop_edges else None,
        )

        # Apply per-frame scale to depths
        for i in range(n_frames):
            all_depths[i] = all_depths[i] * opt_scales[i]

        all_trans = opt_trans
        all_quats = opt_quats

        vprint(
            f'[Any4D-SLAM] PGO done: scale range '
            f'[{opt_scales.min():.4f}, {opt_scales.max():.4f}]'
        )
    else:
        vprint('[Any4D-SLAM] Skipping PGO')

    timing['5_pgo'] = time.time() - t0

    # ---- Step 6: Build compatible NPZ output ----
    t0 = time.time()

    # Build traj array: [N, 7] = [tx, ty, tz, qx, qy, qz, qw]
    # Any4D cam_quats are already [x, y, z, w] which matches the expected format
    traj = np.zeros((n_frames, 7), dtype=np.float32)
    for i in range(n_frames):
        traj[i, :3] = all_trans[i]      # cam2world translation
        traj[i, 3:] = all_quats[i]      # cam2world quaternion [qx, qy, qz, qw]

    # Build disps array: [N, H, W] = 1 / z_depth
    # z_depth = depth_along_ray * ray_dir_z (z-component of unit ray direction)
    disps = np.zeros((n_frames,) + all_depths[0].shape, dtype=np.float32)
    for i in range(n_frames):
        ray_dir_z = all_rays[i][..., 2]  # (H, W)
        z_depth = all_depths[i] * ray_dir_z
        disps[i] = 1.0 / np.maximum(z_depth, 1e-8)

    # scale = 1.0 because Any4D outputs metric-scale depth/poses
    # In load_slam_cam: t_c2w = traj[:, :3] * scale
    # So with scale=1.0, translations are used as-is (already metric)
    scale = 1.0

    # Save
    os.makedirs(f"{seq_folder}/SLAM", exist_ok=True)
    save_path = f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz'
    np.savez(
        save_path,
        tstamp=np.arange(n_frames, dtype=np.int32),
        disps=disps,
        traj=traj,
        img_focal=focal,
        img_center=center,
        scale=scale,
    )
    timing['6_save'] = time.time() - t0

    # Print timing breakdown
    total_time = time.time() - t_start
    timing['total'] = total_time
    video_name = os.path.basename(args.video_path)
    print(f"\n{'='*60}")
    print(f"Any4D-SLAM Stage Timing for {video_name}")
    print(f"{'='*60}")
    for key in ['1_load_model', '2_inference', '3_alignment',
                '4_fusion', '5_pgo', '6_save']:
        t = timing.get(key, 0)
        pct = t / total_time * 100 if total_time > 0 else 0
        print(f"  {key:20s}: {t:7.2f}s ({pct:5.1f}%)")
    print(f"  {'total':20s}: {total_time:7.2f}s")
    print(f"  {'frames':20s}: {n_frames}")
    print(f"  {'chunks':20s}: {len(chunks)}")
    print(f"{'='*60}\n")

    return save_path


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float, default=None)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--chunk_size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--run_pgo", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--any4d_ckpt", type=str, default=None,
                        help="Override Any4D checkpoint path")
    args = parser.parse_args()

    # Need detect_track first to get track indices
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    start_idx, end_idx, _, _ = detect_track_video(args)

    any4d_config = None
    if args.any4d_ckpt:
        any4d_config = {"checkpoint_path": args.any4d_ckpt}

    hawor_slam_any4d(
        args, start_idx, end_idx,
        any4d_config=any4d_config,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        image_size=args.image_size,
        run_pgo=args.run_pgo,
        device=args.device,
    )
