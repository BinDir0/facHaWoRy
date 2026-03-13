"""
Sim(3) Pose Graph Optimization using scipy.optimize.least_squares.

Optimizes global Sim(3) poses (7 DoF per node: 3 rotation + 3 translation + 1 scale)
to minimize residuals from odometry and loop closure edges.
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from typing import List, Tuple, Dict, Optional
import logging

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Sim(3) representation: 7-vector [rx, ry, rz, tx, ty, tz, log_s]
#   - (rx, ry, rz): rotation vector (axis-angle, Rodrigues)
#   - (tx, ty, tz): translation
#   - log_s: log of scale factor (so s = exp(log_s), always positive)
# ---------------------------------------------------------------------------

def sim3_to_matrix(params: np.ndarray) -> np.ndarray:
    """Convert 7-vector Sim(3) params to 4x4 matrix [sR t; 0 1]."""
    rvec = params[:3]
    t = params[3:6]
    s = np.exp(params[6])
    R = Rotation.from_rotvec(rvec).as_matrix()
    M = np.eye(4)
    M[:3, :3] = s * R
    M[:3, 3] = t
    return M


def matrix_to_sim3(M: np.ndarray) -> np.ndarray:
    """Convert 4x4 Sim(3) matrix to 7-vector."""
    sR = M[:3, :3]
    t = M[:3, 3]
    s = np.cbrt(np.linalg.det(sR))  # det(sR) = s^3 * det(R) = s^3
    R = sR / s
    rvec = Rotation.from_matrix(R).as_rotvec()
    return np.array([*rvec, *t, np.log(s)])


def sim3_relative(params_i: np.ndarray, params_j: np.ndarray) -> np.ndarray:
    """Compute relative Sim(3) transform: T_ij = T_i^{-1} @ T_j, returned as 7-vector."""
    Mi = sim3_to_matrix(params_i)
    Mj = sim3_to_matrix(params_j)
    Mi_inv = np.linalg.inv(Mi)
    Mij = Mi_inv @ Mj
    return matrix_to_sim3(Mij)


def sim3_log_error(measured: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    """
    Compute Sim(3) error in tangent space (7-vector).
    Error = Log(T_measured^{-1} @ T_predicted) ≈ predicted - measured for small errors.
    """
    # For small perturbations, the difference in parameter space is a good approximation
    M_meas = sim3_to_matrix(measured)
    M_pred = sim3_to_matrix(predicted)
    M_err = np.linalg.inv(M_meas) @ M_pred
    return matrix_to_sim3(M_err)


# ---------------------------------------------------------------------------
# Edge definitions
# ---------------------------------------------------------------------------

class PGOEdge:
    """A single edge in the Sim(3) pose graph."""
    __slots__ = ("i", "j", "measurement", "sqrt_info")

    def __init__(
        self,
        i: int,
        j: int,
        measurement: np.ndarray,
        information: np.ndarray,
    ):
        """
        Args:
            i, j: node indices
            measurement: 7-vector Sim(3) relative transform T_ij
            information: 7x7 information matrix (inverse covariance)
        """
        self.i = i
        self.j = j
        self.measurement = measurement
        # Precompute sqrt of information for weighted residuals
        eigvals, eigvecs = np.linalg.eigh(information)
        eigvals = np.maximum(eigvals, 0)  # numerical safety
        self.sqrt_info = eigvecs @ np.diag(np.sqrt(eigvals)) @ eigvecs.T


def make_information_matrix(
    sigma_rot: float = 0.01,
    sigma_trans: float = 0.1,
    sigma_scale: float = 0.03,
) -> np.ndarray:
    """Build a diagonal 7x7 information matrix from per-component sigmas."""
    sigmas = np.array([
        sigma_rot, sigma_rot, sigma_rot,
        sigma_trans, sigma_trans, sigma_trans,
        sigma_scale,
    ])
    return np.diag(1.0 / (sigmas ** 2))


# ---------------------------------------------------------------------------
# Pose Graph Optimizer
# ---------------------------------------------------------------------------

class Sim3PoseGraphOptimizer:
    """
    Sim(3) pose graph optimizer using scipy least_squares.

    Nodes: N Sim(3) poses (7 params each).
    Edges: odometry + loop closure constraints.
    Prior: strong prior on node 0 to prevent scale collapse.
    """

    def __init__(self, n_nodes: int):
        self.n_nodes = n_nodes
        self.edges: List[PGOEdge] = []
        self.prior_node: int = 0
        self.prior_params: np.ndarray = np.zeros(7)  # identity pose, s=1
        self.prior_weight: float = 1e3  # strong prior

    def add_odometry_edge(
        self,
        i: int,
        j: int,
        measurement: np.ndarray,
        sigma_rot: float = 0.01,
        sigma_trans: float = 0.1,
        sigma_scale: float = 0.03,
    ):
        """Add an odometry edge between consecutive frames."""
        info = make_information_matrix(sigma_rot, sigma_trans, sigma_scale)
        self.edges.append(PGOEdge(i, j, measurement, info))

    def add_loop_closure_edge(
        self,
        i: int,
        j: int,
        measurement: np.ndarray,
        inlier_ratio: float = 0.5,
        base_sigma_rot: float = 0.05,
        base_sigma_trans: float = 0.5,
        base_sigma_scale: float = 0.1,
    ):
        """Add a loop closure edge with variance scaled by inlier ratio."""
        # Lower inlier ratio → higher variance (less trust)
        scale_factor = 1.0 / max(inlier_ratio, 0.1)
        info = make_information_matrix(
            base_sigma_rot * scale_factor,
            base_sigma_trans * scale_factor,
            base_sigma_scale * scale_factor,
        )
        self.edges.append(PGOEdge(i, j, measurement, info))

    def _residuals(self, x: np.ndarray) -> np.ndarray:
        """Compute all residuals for least_squares."""
        poses = x.reshape(self.n_nodes, 7)
        residuals = []

        # Prior on node 0
        prior_err = (poses[self.prior_node] - self.prior_params) * self.prior_weight
        residuals.append(prior_err)

        # Edge residuals
        for edge in self.edges:
            predicted = sim3_relative(poses[edge.i], poses[edge.j])
            err = sim3_log_error(edge.measurement, predicted)
            weighted_err = edge.sqrt_info @ err
            residuals.append(weighted_err)

        return np.concatenate(residuals)

    def optimize(
        self,
        initial_poses: np.ndarray,
        max_iterations: int = 50,
        verbose: bool = True,
    ) -> np.ndarray:
        """
        Run Sim(3) PGO.

        Args:
            initial_poses: (N, 7) initial Sim(3) params per node
            max_iterations: max LM iterations

        Returns:
            optimized_poses: (N, 7) optimized Sim(3) params
        """
        assert initial_poses.shape == (self.n_nodes, 7)

        x0 = initial_poses.ravel()

        if verbose:
            logger.info(
                f"[PGO] {self.n_nodes} nodes, {len(self.edges)} edges, "
                f"max_iter={max_iterations}"
            )

        result = least_squares(
            self._residuals,
            x0,
            method="lm",
            max_nfev=max_iterations,
            verbose=2 if verbose else 0,
        )

        if verbose:
            logger.info(f"[PGO] cost: {result.cost:.6f}, nfev: {result.nfev}")

        return result.x.reshape(self.n_nodes, 7)


# ---------------------------------------------------------------------------
# High-level API
# ---------------------------------------------------------------------------

def poses_to_sim3_params(
    cam_trans_list: List[np.ndarray],
    cam_quats_list: List[np.ndarray],
) -> np.ndarray:
    """
    Convert lists of cam2world (trans, quat) to (N, 7) Sim(3) params.
    Scale is initialized to 1.0 (log_s = 0).

    Args:
        cam_trans_list: list of (3,) translations
        cam_quats_list: list of (4,) quaternions (x, y, z, w)

    Returns:
        params: (N, 7) array of [rvec, trans, log_s]
    """
    from .sim3_alignment import _quat_to_rotmat
    n = len(cam_trans_list)
    params = np.zeros((n, 7))
    for i in range(n):
        R = _quat_to_rotmat(cam_quats_list[i])
        rvec = Rotation.from_matrix(R).as_rotvec()
        params[i, :3] = rvec
        params[i, 3:6] = cam_trans_list[i]
        params[i, 6] = 0.0  # log(1.0) = 0
    return params


def sim3_params_to_poses(
    params: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Convert (N, 7) Sim(3) params back to cam2world poses + scale factors.

    Returns:
        cam_trans_list: list of (3,) translations
        cam_quats_list: list of (4,) quaternions (x, y, z, w)
        scales: (N,) scale factors
    """
    from .sim3_alignment import _rotmat_to_quat
    n = params.shape[0]
    trans_list = []
    quats_list = []
    scales = np.exp(params[:, 6])

    for i in range(n):
        R = Rotation.from_rotvec(params[i, :3]).as_matrix()
        trans_list.append(params[i, 3:6].copy())
        quats_list.append(_rotmat_to_quat(R))

    return trans_list, quats_list, scales


def build_and_run_pgo(
    cam_trans_list: List[np.ndarray],
    cam_quats_list: List[np.ndarray],
    loop_closure_edges: Optional[List[Dict]] = None,
    odom_sigma_rot: float = 0.01,
    odom_sigma_trans: float = 0.1,
    odom_sigma_scale: float = 0.03,
    max_iterations: int = 50,
) -> Tuple[List[np.ndarray], List[np.ndarray], np.ndarray]:
    """
    Build a Sim(3) pose graph from aligned poses and optimize.

    Args:
        cam_trans_list: list of (3,) aligned translations (all frames)
        cam_quats_list: list of (4,) aligned quaternions (all frames)
        loop_closure_edges: optional list of dicts with keys:
            i, j, R (3x3), t (3,), s (float), inlier_ratio (float)
        odom_sigma_*: odometry edge noise parameters

    Returns:
        opt_trans: optimized translations
        opt_quats: optimized quaternions
        opt_scales: per-frame scale factors (for depth correction)
    """
    n = len(cam_trans_list)
    initial_params = poses_to_sim3_params(cam_trans_list, cam_quats_list)

    optimizer = Sim3PoseGraphOptimizer(n)

    # Add odometry edges (consecutive frames)
    for i in range(n - 1):
        measurement = sim3_relative(initial_params[i], initial_params[i + 1])
        optimizer.add_odometry_edge(
            i, i + 1, measurement,
            sigma_rot=odom_sigma_rot,
            sigma_trans=odom_sigma_trans,
            sigma_scale=odom_sigma_scale,
        )

    # Add loop closure edges
    if loop_closure_edges:
        for lc in loop_closure_edges:
            # Convert R, t, s to Sim(3) 7-vector measurement
            rvec = Rotation.from_matrix(lc["R"]).as_rotvec()
            meas = np.array([*rvec, *lc["t"], np.log(lc["s"])])
            optimizer.add_loop_closure_edge(
                lc["i"], lc["j"], meas,
                inlier_ratio=lc.get("inlier_ratio", 0.5),
            )

    logger.info(f"[PGO] Running with {n} nodes, {len(optimizer.edges)} edges")
    optimized_params = optimizer.optimize(initial_params, max_iterations=max_iterations)

    return sim3_params_to_poses(optimized_params)
