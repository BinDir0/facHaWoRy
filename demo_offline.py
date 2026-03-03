#!/usr/bin/env python3
"""
Offline visualization for HaWoR - no OpenGL/display required.

Uses pyrender with OSMesa backend for pure CPU rendering.
Generates mp4 video directly without requiring display server.

Installation:
    pip install pyrender trimesh
    # For OSMesa (CPU rendering):
    conda install -c conda-forge osmesa
    # Or: sudo apt-get install libosmesa6-dev

Usage:
    python demo_offline.py --video_path video.mp4 --vis_mode world
    python demo_offline.py --video_path video.mp4 --vis_mode cam
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
from tqdm import tqdm

# Set OSMesa backend BEFORE importing pyrender
os.environ['PYOPENGL_PLATFORM'] = 'osmesa'

sys.path.insert(0, os.path.dirname(__file__))

from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_slam import hawor_slam
from scripts.scripts_test_video.hawor_video import hawor_infiller, hawor_motion_estimation


def render_frame_pyrender(vertices_left, vertices_right, faces_left, faces_right,
                          bg_image, camera_pose, focal_length, width, height):
    """Render a single frame using pyrender with OSMesa (CPU)."""
    try:
        import pyrender
        import trimesh
    except ImportError:
        print("Error: pyrender not installed. Install with:")
        print("  pip install pyrender trimesh")
        print("For CPU rendering, also install:")
        print("  conda install -c conda-forge osmesa")
        print("  # Or: sudo apt-get install libosmesa6-dev")
        sys.exit(1)

    # Create scene
    scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3], bg_color=[0, 0, 0, 0])

    # Add left hand mesh (purple)
    if vertices_left is not None and len(vertices_left) > 0:
        mesh_left = trimesh.Trimesh(vertices=vertices_left, faces=faces_left, process=False)
        mesh_left.visual.vertex_colors = [128, 100, 200, 255]
        mesh_pyrender_left = pyrender.Mesh.from_trimesh(mesh_left, smooth=True)
        scene.add(mesh_pyrender_left)

    # Add right hand mesh (blue)
    if vertices_right is not None and len(vertices_right) > 0:
        mesh_right = trimesh.Trimesh(vertices=vertices_right, faces=faces_right, process=False)
        mesh_right.visual.vertex_colors = [53, 152, 202, 255]
        mesh_pyrender_right = pyrender.Mesh.from_trimesh(mesh_right, smooth=True)
        scene.add(mesh_pyrender_right)

    # Add camera
    camera = pyrender.IntrinsicsCamera(
        fx=focal_length, fy=focal_length,
        cx=width / 2, cy=height / 2
    )
    scene.add(camera, pose=camera_pose)

    # Add lights
    light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
    scene.add(light, pose=camera_pose)

    # Render
    renderer = pyrender.OffscreenRenderer(width, height)
    color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    renderer.delete()

    # Composite with background
    if bg_image is not None:
        alpha = color[:, :, 3:4].astype(float) / 255.0
        result = (color[:, :, :3] * alpha + bg_image * (1 - alpha)).astype(np.uint8)
    else:
        result = color[:, :, :3]

    return result
        else:
            result = color

        return result

def create_video_world_mode(left_verts, right_verts, faces_left, faces_right,
                            image_paths, R_c2w, t_c2w, focal_length,
                            output_path, fps=30):
    """Create video in world coordinate mode."""
    print("Rendering video in world mode...")

    # Get video dimensions
    first_img = cv2.imread(image_paths[0])
    height, width = first_img.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    num_frames = left_verts.shape[0]
    for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
        # Load background image
        bg_img = cv2.imread(image_paths[frame_idx])
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

        # Camera pose for this frame
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R_c2w[frame_idx]
        camera_pose[:3, 3] = t_c2w[frame_idx]

        # Render frame
        frame = render_frame_pyrender(
            left_verts[frame_idx], right_verts[frame_idx],
            faces_left, faces_right,
            bg_img, camera_pose,
            focal_length, width, height
        )

        # Write to video
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"✓ Video saved to: {output_path}")
    return output_path


def create_video_cam_mode(left_verts, right_verts, faces_left, faces_right,
                         image_paths, R_w2c, t_w2c, focal_length,
                         output_path, fps=30):
    """Create video in camera coordinate mode."""
    print("Rendering video in camera mode...")

    # Get video dimensions
    first_img = cv2.imread(image_paths[0])
    height, width = first_img.shape[:2]

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    num_frames = left_verts.shape[0]
    for frame_idx in tqdm(range(num_frames), desc="Rendering frames"):
        # Load background image
        bg_img = cv2.imread(image_paths[frame_idx])
        bg_img = cv2.cvtColor(bg_img, cv2.COLOR_BGR2RGB)

        # Camera pose for this frame (world to camera)
        camera_pose = np.eye(4)
        camera_pose[:3, :3] = R_w2c[frame_idx]
        camera_pose[:3, 3] = t_w2c[frame_idx]

        # Render frame
        frame = render_frame_pyrender(
            left_verts[frame_idx], right_verts[frame_idx],
            faces_left, faces_right,
            bg_img, camera_pose,
            focal_length, width, height
        )

        # Write to video
        frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        out.write(frame_bgr)

    out.release()
    print(f"✓ Video saved to: {output_path}")
    return output_path


def main():
    parser = argparse.ArgumentParser(description="Offline HaWoR visualization")
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint", type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight", type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--vis_mode", type=str, default='world', choices=['world', 'cam'],
                       help='Visualization mode: world or cam')
    parser.add_argument("--fps", type=int, default=30, help='FPS for video output')
    args = parser.parse_args()

    # Run inference pipeline (reuse existing results if available)
    print("=== Running HaWoR inference ===")
    start_idx, end_idx, seq_folder, imgfiles = detect_track_video(args)
    frame_chunks_all, img_focal = hawor_motion_estimation(args, start_idx, end_idx, seq_folder)

    slam_path = os.path.join(seq_folder, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        hawor_slam(args, start_idx, end_idx)

    R_w2c_sla_all, t_w2c_sla_all, R_c2w_sla_all, t_c2w_sla_all = load_slam_cam(slam_path)
    pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = hawor_infiller(
        args, start_idx, end_idx, frame_chunks_all
    )

    # Prepare meshes
    print("\n=== Preparing hand meshes ===")
    vis_start = 0
    vis_end = pred_trans.shape[1] - 1

    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234], [234, 38, 239], [38, 122, 239],
                         [239, 122, 279], [122, 118, 279], [279, 118, 215],
                         [118, 117, 215], [215, 117, 214], [117, 119, 214],
                         [214, 119, 121], [119, 120, 121], [121, 120, 78],
                         [120, 108, 78], [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    faces_left = faces_right[:, [0, 2, 1]]

    # Get right hand vertices
    pred_glob_r = run_mano(
        pred_trans[1:2, vis_start:vis_end],
        pred_rot[1:2, vis_start:vis_end],
        pred_hand_pose[1:2, vis_start:vis_end],
        betas=pred_betas[1:2, vis_start:vis_end]
    )
    right_verts = pred_glob_r['vertices'][0].cpu().numpy()

    # Get left hand vertices
    pred_glob_l = run_mano_left(
        pred_trans[0:1, vis_start:vis_end],
        pred_rot[0:1, vis_start:vis_end],
        pred_hand_pose[0:1, vis_start:vis_end],
        betas=pred_betas[0:1, vis_start:vis_end]
    )
    left_verts = pred_glob_l['vertices'][0].cpu().numpy()

    # Transform coordinates
    R_x = torch.tensor([[1, 0, 0], [0, -1, 0], [0, 0, -1]]).float()
    R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)

    left_verts = np.einsum('ij,tnj->tni', R_x.numpy(), left_verts)
    right_verts = np.einsum('ij,tnj->tni', R_x.numpy(), right_verts)

    # Output path
    output_dir = Path(seq_folder) / f"vis_{args.vis_mode}_{vis_start}_{vis_end}"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_video = output_dir / "visualization.mp4"

    # Image paths
    image_names = [str(f) for f in imgfiles[vis_start:vis_end]]

    # Render video
    print(f"\n=== Rendering video ({args.vis_mode} mode) ===")
    if args.vis_mode == 'world':
        create_video_world_mode(
            left_verts, right_verts, faces_left, faces_right,
            image_names, R_c2w_sla_all[vis_start:vis_end].cpu().numpy(),
            t_c2w_sla_all[vis_start:vis_end].cpu().numpy(),
            img_focal, output_video, fps=args.fps
        )
    else:  # cam mode
        create_video_cam_mode(
            left_verts, right_verts, faces_left, faces_right,
            image_names, R_w2c_sla_all[vis_start:vis_end].cpu().numpy(),
            t_w2c_sla_all[vis_start:vis_end].cpu().numpy(),
            img_focal, output_video, fps=args.fps
        )

    print(f"\n✓ Done! Video saved to: {output_video}")
    print(f"\nTo view: scp user@server:{output_video} ./")


if __name__ == '__main__':
    main()
