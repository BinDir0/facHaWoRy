#!/usr/bin/env python3
"""
Test Any4D SLAM pipeline on a single video and generate visualization.

Usage:
    python scripts/test_any4d_slam_viz.py \
        --video_list videos.txt \
        --k 0 \
        --output_dir ./viz_output
"""

import os
import sys
import time
import argparse
import subprocess
import numpy as np
from pathlib import Path

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.transform import Rotation


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def read_video_list(path):
    with open(path, 'r') as f:
        return [line.strip() for line in f if line.strip()]


def ensure_frames_extracted(video_path):
    """Extract frames if not already done."""
    stem = Path(video_path).stem
    extracted_dir = Path(video_path).parent / stem / "extracted_images"
    if extracted_dir.exists() and len(list(extracted_dir.glob("*.jpg"))) > 0:
        print(f"[viz] Frames already extracted: {extracted_dir}")
        return
    print(f"[viz] Extracting frames from {video_path} ...")
    script = os.path.join(os.path.dirname(__file__), "extract_frames.py")
    subprocess.check_call([sys.executable, script, "--video_path", video_path])


def quat_to_rotmat(q):
    """Quaternion [qx, qy, qz, qw] -> 3x3 rotation matrix."""
    return Rotation.from_quat(q).as_matrix()


def build_intrinsic(focal, cx, cy):
    K = np.eye(3)
    K[0, 0] = focal
    K[1, 1] = focal
    K[0, 2] = cx
    K[1, 2] = cy
    return K


# ---------------------------------------------------------------------------
# Visualization functions
# ---------------------------------------------------------------------------

def viz_depth_samples(disps, frame_source, output_dir, n_samples=8):
    """Show RGB + depth side-by-side for evenly spaced frames."""
    n_frames = len(disps)
    indices = np.linspace(0, n_frames - 1, n_samples, dtype=int)

    fig, axes = plt.subplots(n_samples, 2, figsize=(12, 3 * n_samples))
    if n_samples == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(indices):
        rgb = frame_source.get_frame(idx, rgb=True)
        depth = 1.0 / np.maximum(disps[idx], 1e-8)

        axes[row, 0].imshow(rgb)
        axes[row, 0].set_title(f"Frame {idx} - RGB")
        axes[row, 0].axis('off')

        im = axes[row, 1].imshow(depth, cmap='viridis')
        axes[row, 1].set_title(f"Frame {idx} - Depth (z)")
        axes[row, 1].axis('off')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(output_dir, "depth_samples.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Saved {path}")


def viz_camera_trajectory(traj, chunk_ranges, output_dir):
    """3D plot of camera positions colored by time."""
    positions = traj[:, :3]
    n = len(positions)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    colors = plt.cm.viridis(np.linspace(0, 1, n))
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
               c=colors, s=4, alpha=0.8)
    ax.plot(positions[:, 0], positions[:, 1], positions[:, 2],
            color='gray', alpha=0.3, linewidth=0.5)

    # Mark first and last
    ax.scatter(*positions[0], color='red', s=80, marker='o', label='Start')
    ax.scatter(*positions[-1], color='blue', s=80, marker='s', label='End')

    # Mark chunk boundaries
    for i, (start, end) in enumerate(chunk_ranges):
        if i > 0:
            ax.scatter(*positions[start], color='orange', s=40, marker='^',
                       zorder=5)

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title('Camera Trajectory (colored by time)')
    ax.legend()

    path = os.path.join(output_dir, "camera_trajectory.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Saved {path}")


def viz_depth_stats(disps, chunk_ranges, output_dir):
    """Per-frame depth statistics with chunk boundaries."""
    n_frames = len(disps)
    means = []
    medians = []
    stds = []

    for i in range(n_frames):
        depth = 1.0 / np.maximum(disps[i], 1e-8)
        # Clip extreme values for statistics
        valid = depth[depth < np.percentile(depth, 99)]
        means.append(np.mean(valid))
        medians.append(np.median(valid))
        stds.append(np.std(valid))

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

    frames = np.arange(n_frames)
    ax1.plot(frames, means, label='Mean depth', linewidth=0.8)
    ax1.plot(frames, medians, label='Median depth', linewidth=0.8)
    ax1.set_ylabel('Depth (m)')
    ax1.set_title('Per-frame Depth Statistics')
    ax1.legend()

    ax2.plot(frames, stds, label='Std dev', linewidth=0.8, color='orange')
    ax2.set_xlabel('Frame index')
    ax2.set_ylabel('Depth std (m)')
    ax2.legend()

    # Mark chunk boundaries on both axes
    for ax in [ax1, ax2]:
        for i, (start, end) in enumerate(chunk_ranges):
            if i > 0:
                ax.axvline(x=start, color='red', linestyle='--',
                           alpha=0.5, linewidth=0.8)
        # Mark overlap midpoints (where hard switch happens)
        for i in range(1, len(chunk_ranges)):
            prev_end = chunk_ranges[i - 1][1]
            curr_start = chunk_ranges[i][0]
            mid = (curr_start + prev_end) // 2
            ax.axvline(x=mid, color='green', linestyle=':',
                       alpha=0.5, linewidth=0.8)

    plt.tight_layout()
    path = os.path.join(output_dir, "depth_stats.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f"[viz] Saved {path}")


def viz_reprojection(disps, traj, focal, center, frame_source, output_dir,
                     n_pairs=6):
    """Reprojection error between adjacent frame pairs."""
    n_frames = len(disps)
    if n_frames < 2:
        print("[viz] Not enough frames for reprojection check")
        return

    # Pick evenly spaced pairs
    gap = max(1, n_frames // (n_pairs + 1))
    pair_indices = [(i, i + 1) for i in range(gap, n_frames - 1, gap)]
    pair_indices = pair_indices[:n_pairs]

    K = build_intrinsic(focal, center[0], center[1])
    K_inv = np.linalg.inv(K)

    errors_all = []

    fig, axes = plt.subplots(len(pair_indices), 2, figsize=(14, 4 * len(pair_indices)))
    if len(pair_indices) == 1:
        axes = axes[np.newaxis, :]

    for row, (i, j) in enumerate(pair_indices):
        # Frame i depth
        depth_i = 1.0 / np.maximum(disps[i], 1e-8)  # [H, W]
        H, W = depth_i.shape

        # Camera poses (cam2world)
        R_i = quat_to_rotmat(traj[i, 3:7])
        t_i = traj[i, :3]
        R_j = quat_to_rotmat(traj[j, 3:7])
        t_j = traj[j, :3]

        # World2cam for frame j
        R_j_inv = R_j.T
        t_j_inv = -R_j_inv @ t_j

        # Build pixel grid
        v, u = np.meshgrid(np.arange(H), np.arange(W), indexing='ij')
        uv1 = np.stack([u, v, np.ones_like(u)], axis=-1).astype(np.float64)  # [H,W,3]

        # Unproject frame i pixels to 3D (camera frame)
        rays_cam = (K_inv @ uv1.reshape(-1, 3).T).T.reshape(H, W, 3)
        pts_cam_i = rays_cam * depth_i[..., np.newaxis]  # [H,W,3]

        # Transform to world
        pts_world = (R_i @ pts_cam_i.reshape(-1, 3).T).T + t_i  # [N,3]

        # Project to frame j
        pts_cam_j = (R_j_inv @ pts_world.T).T + t_j_inv  # [N,3]
        pts_cam_j = pts_cam_j.reshape(H, W, 3)

        # Project to pixel
        z_j = pts_cam_j[..., 2]
        valid = z_j > 0.01
        proj_uv = np.zeros((H, W, 2))
        proj_uv[..., 0] = focal * pts_cam_j[..., 0] / np.maximum(z_j, 1e-8) + center[0]
        proj_uv[..., 1] = focal * pts_cam_j[..., 1] / np.maximum(z_j, 1e-8) + center[1]

        # Compute pixel error
        err_u = proj_uv[..., 0] - u
        err_v = proj_uv[..., 1] - v
        pixel_err = np.sqrt(err_u ** 2 + err_v ** 2)

        # Mask out invalid and boundary
        valid &= (proj_uv[..., 0] >= 0) & (proj_uv[..., 0] < W)
        valid &= (proj_uv[..., 1] >= 0) & (proj_uv[..., 1] < H)
        pixel_err[~valid] = np.nan

        mean_err = np.nanmean(pixel_err)
        median_err = np.nanmedian(pixel_err)
        errors_all.append(mean_err)

        # Plot RGB + error heatmap
        rgb_i = frame_source.get_frame(i, rgb=True)
        axes[row, 0].imshow(rgb_i)
        axes[row, 0].set_title(f"Frame {i}")
        axes[row, 0].axis('off')

        im = axes[row, 1].imshow(pixel_err, cmap='hot', vmin=0,
                                  vmax=min(10, np.nanpercentile(pixel_err, 95)))
        axes[row, 1].set_title(
            f"Reproj err {i}→{j}: mean={mean_err:.2f}px, med={median_err:.2f}px")
        axes[row, 1].axis('off')
        plt.colorbar(im, ax=axes[row, 1], fraction=0.046, pad=0.04)

    plt.tight_layout()
    path = os.path.join(output_dir, "reprojection_check.png")
    fig.savefig(path, dpi=120, bbox_inches='tight')
    plt.close(fig)

    avg_err = np.nanmean(errors_all)
    print(f"[viz] Saved {path}")
    print(f"[viz] Average reprojection error: {avg_err:.2f} px")
    return avg_err


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Test Any4D SLAM on a single video with visualization")
    parser.add_argument("--video_list", type=str, required=True,
                        help="Text file with one .mp4 path per line")
    parser.add_argument("--k", type=int, default=0,
                        help="Index of video to process (default: 0)")
    parser.add_argument("--chunk_size", type=int, default=180)
    parser.add_argument("--overlap", type=int, default=30)
    parser.add_argument("--image_size", type=int, default=518)
    parser.add_argument("--run_pgo", action="store_true")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_dir", type=str, default="./viz_output",
                        help="Directory for visualization output")
    parser.add_argument("--any4d_ckpt", type=str, default=None)
    args = parser.parse_args()

    # 1. Read video list, pick k-th video
    videos = read_video_list(args.video_list)
    if args.k >= len(videos):
        print(f"ERROR: k={args.k} but only {len(videos)} videos in list",
              file=sys.stderr)
        sys.exit(1)
    video_path = videos[args.k]
    print(f"[viz] Selected video [{args.k}]: {video_path}")

    if not os.path.isfile(video_path):
        print(f"ERROR: video file not found: {video_path}", file=sys.stderr)
        sys.exit(1)

    # 2. Extract frames
    ensure_frames_extracted(video_path)

    # 3. Run Any4D SLAM
    from scripts.scripts_test_video.hawor_slam_any4d import hawor_slam_any4d

    # Count frames to get end_idx
    from lib.pipeline.frame_source import build_frame_source
    frame_source = build_frame_source(video_path)
    n_frames = len(frame_source)
    print(f"[viz] Total frames: {n_frames}")

    # Build minimal args namespace
    slam_args = argparse.Namespace(
        video_path=video_path,
        img_focal=None,
    )

    any4d_config = None
    if args.any4d_ckpt:
        any4d_config = {"checkpoint_path": args.any4d_ckpt}

    print(f"\n{'='*60}")
    print(f"Running Any4D SLAM: chunk_size={args.chunk_size}, "
          f"overlap={args.overlap}, pgo={args.run_pgo}")
    print(f"{'='*60}\n")

    t0 = time.time()
    npz_path = hawor_slam_any4d(
        slam_args,
        start_idx=0,
        end_idx=n_frames,
        any4d_config=any4d_config,
        chunk_size=args.chunk_size,
        overlap=args.overlap,
        image_size=args.image_size,
        run_pgo=args.run_pgo,
        device=args.device,
    )
    slam_time = time.time() - t0
    print(f"[viz] SLAM finished in {slam_time:.1f}s → {npz_path}")

    # 4. Load NPZ and visualize
    data = np.load(npz_path)
    disps = data['disps']       # [N, H, W]
    traj = data['traj']         # [N, 7]
    focal = float(data['img_focal'])
    center = data['img_center']  # [2]

    os.makedirs(args.output_dir, exist_ok=True)

    # Compute chunk ranges for visualization
    stride = args.chunk_size - args.overlap
    chunk_ranges = []
    start = 0
    while start < n_frames:
        end = min(start + args.chunk_size, n_frames)
        chunk_ranges.append((start, end))
        start += stride
        if end == n_frames:
            break

    print(f"\n[viz] Generating visualizations → {args.output_dir}/")

    # a) Depth samples
    viz_depth_samples(disps, frame_source, args.output_dir)

    # b) Camera trajectory
    viz_camera_trajectory(traj, chunk_ranges, args.output_dir)

    # c) Depth statistics
    viz_depth_stats(disps, chunk_ranges, args.output_dir)

    # d) Reprojection check
    avg_reproj = viz_reprojection(
        disps, traj, focal, center, frame_source, args.output_dir)

    # 5. Summary
    print(f"\n{'='*60}")
    print(f"Summary")
    print(f"{'='*60}")
    print(f"  Video:             {video_path}")
    print(f"  Frames:            {n_frames}")
    print(f"  Chunks:            {len(chunk_ranges)}")
    print(f"  SLAM time:         {slam_time:.1f}s ({slam_time/n_frames:.2f}s/frame)")
    print(f"  Depth shape:       {disps.shape}")
    print(f"  Focal:             {focal:.1f}")
    if avg_reproj is not None:
        print(f"  Avg reproj error:  {avg_reproj:.2f} px")
    print(f"  Output dir:        {os.path.abspath(args.output_dir)}")
    print(f"  NPZ path:          {npz_path}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
