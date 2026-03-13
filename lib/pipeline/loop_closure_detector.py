"""
Loop closure detection for Sim(3) pose graph optimization.

Detects co-visible frame pairs using feature matching (LightGlue),
lifts 2D matches to 3D using depth, and estimates Sim(3) constraints
via RANSAC-Umeyama.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def detect_loop_candidates(
    cam_trans_list: List[np.ndarray],
    min_frame_gap: int = 50,
    max_spatial_distance: float = 2.0,
    max_candidates: int = 20,
) -> List[Tuple[int, int]]:
    """
    Find candidate loop closure pairs: frames that are far apart temporally
    but close spatially (likely co-visible).

    Args:
        cam_trans_list: list of (3,) camera translations
        min_frame_gap: minimum frame index difference
        max_spatial_distance: max Euclidean distance between cameras
        max_candidates: maximum number of candidates to return

    Returns:
        candidates: list of (frame_i, frame_j) pairs
    """
    n = len(cam_trans_list)
    positions = np.array(cam_trans_list)  # (N, 3)

    candidates = []
    # Check all pairs with sufficient temporal gap
    for i in range(n):
        for j in range(i + min_frame_gap, n):
            dist = np.linalg.norm(positions[i] - positions[j])
            if dist < max_spatial_distance:
                candidates.append((i, j, dist))

    # Sort by spatial distance (closest first) and take top-k
    candidates.sort(key=lambda x: x[2])
    return [(c[0], c[1]) for c in candidates[:max_candidates]]


def estimate_sim3_from_depth_correspondences(
    pts3d_i: np.ndarray,
    pts3d_j: np.ndarray,
    max_iterations: int = 500,
    inlier_threshold: float = 0.1,
) -> Optional[Dict]:
    """
    Estimate Sim(3) from 3D-3D correspondences using RANSAC-Umeyama.

    Args:
        pts3d_i: (M, 3) 3D points from frame i (world frame)
        pts3d_j: (M, 3) 3D points from frame j (world frame)

    Returns:
        dict with R, t, s, inlier_ratio, or None if failed
    """
    from .sim3_alignment import ransac_umeyama_sim3

    if len(pts3d_i) < 4:
        return None

    try:
        R, t, s, inlier_mask = ransac_umeyama_sim3(
            pts3d_j, pts3d_i,
            max_iterations=max_iterations,
            inlier_threshold=inlier_threshold,
            min_inlier_ratio=0.2,
        )
    except RuntimeError:
        return None

    inlier_ratio = inlier_mask.sum() / len(inlier_mask)
    return {"R": R, "t": t, "s": s, "inlier_ratio": inlier_ratio}


def compute_loop_closure_edges(
    cam_trans_list: List[np.ndarray],
    cam_quats_list: List[np.ndarray],
    depth_maps: Dict[int, np.ndarray],
    ray_dirs_list: List[np.ndarray],
    scene_flow_list: Optional[List[np.ndarray]] = None,
    min_frame_gap: int = 50,
    max_spatial_distance: float = 2.0,
    max_candidates: int = 20,
    flow_threshold: float = 0.02,
    pixel_stride: int = 8,
) -> List[Dict]:
    """
    Detect loop closures and compute Sim(3) constraints.

    This uses depth-based 3D point matching (no LightGlue needed for
    the initial version — we use pixel-wise correspondences from
    co-visible static regions).

    Args:
        cam_trans_list: aligned translations
        cam_quats_list: aligned quaternions
        depth_maps: dict frame_idx -> (H, W) depth
        ray_dirs_list: list of (H, W, 3) ray directions
        scene_flow_list: optional, for static filtering
        min_frame_gap: minimum temporal gap for loop candidates
        max_spatial_distance: max camera distance for candidates
        flow_threshold: scene flow threshold for static pixels
        pixel_stride: spatial subsampling

    Returns:
        edges: list of dicts {i, j, R, t, s, inlier_ratio}
    """
    from .sim3_alignment import backproject_to_pointcloud, filter_static_pixels

    candidates = detect_loop_candidates(
        cam_trans_list, min_frame_gap, max_spatial_distance, max_candidates,
    )

    if not candidates:
        logger.info("[LoopClosure] No candidates found")
        return []

    logger.info(f"[LoopClosure] Testing {len(candidates)} candidates")
    edges = []

    for frame_i, frame_j in candidates:
        if frame_i not in depth_maps or frame_j not in depth_maps:
            continue

        depth_i = depth_maps[frame_i]
        depth_j = depth_maps[frame_j]

        # Backproject both frames
        pts_i = backproject_to_pointcloud(
            depth_i, ray_dirs_list[frame_i],
            cam_trans_list[frame_i], cam_quats_list[frame_i],
        )
        pts_j = backproject_to_pointcloud(
            depth_j, ray_dirs_list[frame_j],
            cam_trans_list[frame_j], cam_quats_list[frame_j],
        )

        H, W = pts_i.shape[:2]

        # Static mask
        if scene_flow_list is not None:
            mask_i = filter_static_pixels(scene_flow_list[frame_i], flow_threshold)
            mask_j = filter_static_pixels(scene_flow_list[frame_j], flow_threshold)
            static_mask = mask_i & mask_j
        else:
            static_mask = np.ones((H, W), dtype=bool)

        # Valid depth mask
        valid = (depth_i > 1e-6) & (depth_j > 1e-6) & static_mask

        # Subsample
        sub = np.zeros((H, W), dtype=bool)
        sub[::pixel_stride, ::pixel_stride] = True
        valid &= sub

        pts_i_flat = pts_i[valid]
        pts_j_flat = pts_j[valid]

        if len(pts_i_flat) < 10:
            continue

        result = estimate_sim3_from_depth_correspondences(
            pts_i_flat, pts_j_flat,
        )

        if result is not None and result["inlier_ratio"] > 0.3:
            result["i"] = frame_i
            result["j"] = frame_j
            edges.append(result)
            logger.info(
                f"[LoopClosure] {frame_i}↔{frame_j}: "
                f"s={result['s']:.4f}, inlier={result['inlier_ratio']:.2%}"
            )

    logger.info(f"[LoopClosure] Found {len(edges)} valid loop closures")
    return edges
