"""
Sim(3) chunk alignment for Any4D multi-chunk video processing.

Aligns overlapping chunks using RANSAC-Umeyama on static 3D point clouds,
then scales depth maps to maintain geometric consistency.
"""

import numpy as np
from typing import Tuple, Optional, Dict, List


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute Sim(3) transformation (R, t, s) that maps src -> dst.
    dst = s * R @ src + t

    Uses Umeyama's algorithm (closed-form least-squares solution).

    Args:
        src: (N, 3) source points
        dst: (N, 3) destination points

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scalar scale factor
    """
    assert src.shape == dst.shape and src.shape[1] == 3
    n = src.shape[0]

    # Centroids
    mu_src = src.mean(axis=0)
    mu_dst = dst.mean(axis=0)

    # Centered points
    src_c = src - mu_src
    dst_c = dst - mu_dst

    # Variance of source
    var_src = np.sum(src_c ** 2) / n

    # Cross-covariance matrix
    cov = (dst_c.T @ src_c) / n

    # SVD
    U, D, Vt = np.linalg.svd(cov)

    # Handle reflection
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1

    # Rotation
    R = U @ S @ Vt

    # Scale
    s = np.trace(np.diag(D) @ S) / var_src

    # Translation
    t = mu_dst - s * R @ mu_src

    return R, t, s


def ransac_umeyama_sim3(
    src: np.ndarray,
    dst: np.ndarray,
    max_iterations: int = 1000,
    inlier_threshold: float = 0.05,
    min_inlier_ratio: float = 0.3,
    min_points: int = 4,
) -> Tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """
    RANSAC-based Sim(3) estimation using Umeyama on corresponding 3D point pairs.

    Args:
        src: (N, 3) source points (Chunk B coordinate system)
        dst: (N, 3) destination points (Chunk A coordinate system)
        max_iterations: RANSAC iterations
        inlier_threshold: max distance for a point to be considered inlier
        min_inlier_ratio: minimum fraction of inliers for a valid model
        min_points: minimum points to fit Umeyama (>=3 for Sim3)

    Returns:
        R: (3, 3) rotation matrix
        t: (3,) translation vector
        s: scalar scale factor
        inlier_mask: (N,) boolean mask of inliers
    """
    n = src.shape[0]
    assert n >= min_points, f"Need at least {min_points} points, got {n}"

    best_inliers = None
    best_num_inliers = 0
    best_R, best_t, best_s = None, None, None

    rng = np.random.default_rng(42)

    for _ in range(max_iterations):
        # Random sample
        idx = rng.choice(n, size=min_points, replace=False)
        try:
            R, t, s = umeyama_sim3(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue

        if s <= 0 or not np.isfinite(s):
            continue

        # Compute residuals for all points
        transformed = s * (R @ src.T).T + t
        residuals = np.linalg.norm(transformed - dst, axis=1)
        inlier_mask = residuals < inlier_threshold

        num_inliers = inlier_mask.sum()
        if num_inliers > best_num_inliers:
            best_num_inliers = num_inliers
            best_inliers = inlier_mask
            best_R, best_t, best_s = R, t, s

    if best_inliers is None or best_num_inliers < min_inlier_ratio * n:
        raise RuntimeError(
            f"RANSAC failed: best {best_num_inliers}/{n} inliers "
            f"(need {min_inlier_ratio:.0%})"
        )

    # Refine on all inliers
    R, t, s = umeyama_sim3(src[best_inliers], dst[best_inliers])

    # Recompute inliers with refined model
    transformed = s * (R @ src.T).T + t
    residuals = np.linalg.norm(transformed - dst, axis=1)
    inlier_mask = residuals < inlier_threshold

    return R, t, s, inlier_mask


# ---------------------------------------------------------------------------
# Chunk alignment utilities
# ---------------------------------------------------------------------------

def backproject_to_pointcloud(
    depth_along_ray: np.ndarray,
    ray_dirs: np.ndarray,
    cam_trans: np.ndarray,
    cam_quats: np.ndarray,
) -> np.ndarray:
    """
    Backproject depth map to world-frame 3D point cloud.

    Args:
        depth_along_ray: (H, W) or (H, W, 1) — Euclidean distance along ray
        ray_dirs: (H, W, 3) — unit ray directions in camera frame
        cam_trans: (3,) — camera translation (cam2world)
        cam_quats: (4,) — camera quaternion (x, y, z, w) cam2world

    Returns:
        pts_world: (H, W, 3) — 3D points in world frame
    """
    if depth_along_ray.ndim == 3:
        depth_along_ray = depth_along_ray[..., 0]

    # P_cam = depth * ray_dir (ray_dir is unit vector)
    pts_cam = depth_along_ray[..., None] * ray_dirs  # (H, W, 3)

    # Quaternion to rotation matrix
    R = _quat_to_rotmat(cam_quats)

    # P_world = R @ P_cam + t
    H, W = pts_cam.shape[:2]
    pts_flat = pts_cam.reshape(-1, 3)  # (H*W, 3)
    pts_world_flat = (R @ pts_flat.T).T + cam_trans  # (H*W, 3)

    return pts_world_flat.reshape(H, W, 3)


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Quaternion (x, y, z, w) to 3x3 rotation matrix."""
    x, y, z, w = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z),     2*(x*z + w*y)],
        [2*(x*y + w*z),     1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y),     2*(y*z + w*x),     1 - 2*(x*x + y*y)],
    ])


def filter_static_pixels(
    scene_flow: np.ndarray,
    flow_threshold: float = 0.02,
) -> np.ndarray:
    """
    Create a mask of static pixels based on scene flow magnitude.

    Args:
        scene_flow: (H, W, 3) — world-frame 3D motion vectors
        flow_threshold: max flow magnitude to be considered static

    Returns:
        static_mask: (H, W) boolean — True for static pixels
    """
    flow_mag = np.linalg.norm(scene_flow, axis=-1)  # (H, W)
    return flow_mag < flow_threshold


def _build_overlap_pointclouds(
    chunk_a: Dict,
    chunk_b: Dict,
    overlap_frame_indices: List[int],
    sample_frames: int = 3,
    flow_threshold: float = 0.02,
    pixel_stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build corresponding static 3D point pairs from the overlap region.

    Each chunk dict must contain per-frame arrays:
        depth_along_ray: list of (H, W) or (H, W, 1)
        ray_dirs: list of (H, W, 3)
        cam_trans: list of (3,)
        cam_quats: list of (4,)
        scene_flow: list of (H, W, 3) or None

    Args:
        chunk_a: chunk data dict (frames indexed in chunk-local coords)
        chunk_b: chunk data dict
        overlap_frame_indices: global frame indices in the overlap region
        sample_frames: how many frames to sample from overlap (first/mid/last)
        flow_threshold: scene flow threshold for static filtering
        pixel_stride: spatial subsampling stride to reduce point count

    Returns:
        pts_a: (M, 3) points in chunk A's world frame
        pts_b: (M, 3) points in chunk B's world frame
    """
    n_overlap = len(overlap_frame_indices)
    if sample_frames >= n_overlap:
        sampled_indices = list(range(n_overlap))
    else:
        # Sample first, middle, last
        sampled_indices = [
            0,
            n_overlap // 2,
            n_overlap - 1,
        ][:sample_frames]

    all_pts_a = []
    all_pts_b = []

    for local_idx in sampled_indices:
        global_idx = overlap_frame_indices[local_idx]

        # Get local indices within each chunk
        idx_in_a = chunk_a["global_to_local"][global_idx]
        idx_in_b = chunk_b["global_to_local"][global_idx]

        # Backproject in both coordinate systems
        pts_a = backproject_to_pointcloud(
            chunk_a["depth_along_ray"][idx_in_a],
            chunk_a["ray_dirs"][idx_in_a],
            chunk_a["cam_trans"][idx_in_a],
            chunk_a["cam_quats"][idx_in_a],
        )
        pts_b = backproject_to_pointcloud(
            chunk_b["depth_along_ray"][idx_in_b],
            chunk_b["ray_dirs"][idx_in_b],
            chunk_b["cam_trans"][idx_in_b],
            chunk_b["cam_quats"][idx_in_b],
        )

        # Static pixel mask (use chunk A's scene flow if available)
        H, W = pts_a.shape[:2]
        if chunk_a.get("scene_flow") is not None:
            sf_a = chunk_a["scene_flow"][idx_in_a]
            static_mask = filter_static_pixels(sf_a, flow_threshold)
        else:
            static_mask = np.ones((H, W), dtype=bool)

        # Filter out invalid depth
        depth_a = chunk_a["depth_along_ray"][idx_in_a]
        depth_b = chunk_b["depth_along_ray"][idx_in_b]
        if depth_a.ndim == 3:
            depth_a = depth_a[..., 0]
        if depth_b.ndim == 3:
            depth_b = depth_b[..., 0]
        valid_mask = (depth_a > 1e-6) & (depth_b > 1e-6) & static_mask

        # Spatial subsampling
        subsample_mask = np.zeros((H, W), dtype=bool)
        subsample_mask[::pixel_stride, ::pixel_stride] = True
        valid_mask &= subsample_mask

        all_pts_a.append(pts_a[valid_mask])
        all_pts_b.append(pts_b[valid_mask])

    pts_a = np.concatenate(all_pts_a, axis=0)
    pts_b = np.concatenate(all_pts_b, axis=0)

    return pts_a, pts_b


def transform_chunk_poses(
    cam_trans_list: List[np.ndarray],
    cam_quats_list: List[np.ndarray],
    R: np.ndarray,
    t: np.ndarray,
    s: float,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    Apply Sim(3) transform to a list of cam2world poses.
    New pose: T_new = Sim3(R,t,s) ∘ T_old
    Concretely: t_new = s * R @ t_old + t, R_new = R @ R_old

    Returns:
        new_trans: list of (3,) transformed translations
        new_quats: list of (4,) transformed quaternions (x,y,z,w)
    """
    new_trans = []
    new_quats = []
    for ct, cq in zip(cam_trans_list, cam_quats_list):
        R_old = _quat_to_rotmat(cq)
        t_new = s * R @ ct + t
        R_new = R @ R_old
        q_new = _rotmat_to_quat(R_new)
        new_trans.append(t_new)
        new_quats.append(q_new)
    return new_trans, new_quats


def _rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """3x3 rotation matrix to quaternion (x, y, z, w)."""
    trace = np.trace(R)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (R[2, 1] - R[1, 2]) * s
        y = (R[0, 2] - R[2, 0]) * s
        z = (R[1, 0] - R[0, 1]) * s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def align_chunk_pair(
    chunk_a: Dict,
    chunk_b: Dict,
    overlap_frame_indices: List[int],
    ransac_iterations: int = 1000,
    inlier_threshold: float = 0.05,
    flow_threshold: float = 0.02,
    pixel_stride: int = 4,
) -> Tuple[np.ndarray, np.ndarray, float, float]:
    """
    Align Chunk B to Chunk A using Sim(3) from overlap region point clouds.
    Modifies chunk_b in-place: updates poses and scales depths.

    Args:
        chunk_a: already-aligned chunk data
        chunk_b: chunk to be aligned
        overlap_frame_indices: global frame indices in overlap
        ransac_iterations: RANSAC iterations
        inlier_threshold: RANSAC inlier distance threshold
        flow_threshold: scene flow threshold for static filtering
        pixel_stride: spatial subsampling stride

    Returns:
        R, t, s: the Sim(3) transform applied
        inlier_ratio: fraction of inlier points
    """
    # Build corresponding point clouds from overlap
    pts_a, pts_b = _build_overlap_pointclouds(
        chunk_a, chunk_b, overlap_frame_indices,
        flow_threshold=flow_threshold,
        pixel_stride=pixel_stride,
    )

    # Compute Sim(3): pts_a = s * R @ pts_b + t
    R, t, s, inlier_mask = ransac_umeyama_sim3(
        pts_b, pts_a,
        max_iterations=ransac_iterations,
        inlier_threshold=inlier_threshold,
    )
    inlier_ratio = inlier_mask.sum() / len(inlier_mask)

    # Transform chunk B poses
    chunk_b["cam_trans"], chunk_b["cam_quats"] = transform_chunk_poses(
        chunk_b["cam_trans"], chunk_b["cam_quats"], R, t, s,
    )

    # Scale chunk B depths
    for i in range(len(chunk_b["depth_along_ray"])):
        chunk_b["depth_along_ray"][i] = chunk_b["depth_along_ray"][i] * s

    return R, t, s, inlier_ratio


def fuse_overlap_depths(
    chunks: List[Dict],
    chunk_ranges: List[Tuple[int, int]],
    method: str = "hard_switch",
) -> Dict[int, np.ndarray]:
    """
    Fuse depth maps in overlap regions between aligned chunks.

    Args:
        chunks: list of aligned chunk dicts
        chunk_ranges: list of (start_frame, end_frame) for each chunk
        method: "hard_switch" (cut at midpoint) or "alpha_blend"

    Returns:
        fused_depths: dict mapping global_frame_idx -> (H, W) depth array
    """
    fused = {}

    for i, chunk in enumerate(chunks):
        start_i, end_i = chunk_ranges[i]

        for global_idx in range(start_i, end_i):
            local_idx = chunk["global_to_local"][global_idx]
            depth = chunk["depth_along_ray"][local_idx]
            if depth.ndim == 3:
                depth = depth[..., 0]

            if global_idx not in fused:
                fused[global_idx] = depth
            else:
                # Overlap: this frame exists in previous chunk too
                if method == "hard_switch":
                    # Find overlap midpoint with previous chunk
                    prev_start, prev_end = chunk_ranges[i - 1]
                    overlap_mid = (start_i + prev_end) // 2
                    if global_idx >= overlap_mid:
                        fused[global_idx] = depth  # Use current chunk
                    # else: keep previous chunk's depth
                elif method == "alpha_blend":
                    prev_start, prev_end = chunk_ranges[i - 1]
                    overlap_len = prev_end - start_i
                    if overlap_len > 0:
                        alpha = (global_idx - start_i) / overlap_len
                        fused[global_idx] = (
                            (1 - alpha) * fused[global_idx] + alpha * depth
                        )

    return fused


def align_all_chunks(
    chunks: List[Dict],
    chunk_ranges: List[Tuple[int, int]],
    ransac_iterations: int = 1000,
    inlier_threshold: float = 0.05,
    flow_threshold: float = 0.02,
    pixel_stride: int = 4,
    depth_fusion: str = "hard_switch",
) -> Tuple[List[Dict], Dict[int, np.ndarray]]:
    """
    Sequentially align all chunks and fuse overlap depths.

    Args:
        chunks: list of chunk dicts from Any4D inference
        chunk_ranges: list of (start_frame, end_frame) per chunk
        depth_fusion: "hard_switch" or "alpha_blend"

    Returns:
        chunks: aligned chunks (modified in-place)
        fused_depths: dict mapping global_frame_idx -> (H, W) depth
    """
    alignment_results = []

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
            ransac_iterations=ransac_iterations,
            inlier_threshold=inlier_threshold,
            flow_threshold=flow_threshold,
            pixel_stride=pixel_stride,
        )
        alignment_results.append({
            "chunk_pair": (i - 1, i),
            "scale": s,
            "inlier_ratio": inlier_ratio,
        })
        print(
            f"[Sim3] Chunk {i-1}→{i}: s={s:.4f}, "
            f"inlier_ratio={inlier_ratio:.2%}, "
            f"overlap={len(overlap_indices)} frames"
        )

    # Fuse depths
    fused_depths = fuse_overlap_depths(chunks, chunk_ranges, method=depth_fusion)

    return chunks, fused_depths
