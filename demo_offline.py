#!/usr/bin/env python3
"""
Offline visualization for HaWoR - no OpenGL/display required.

This script saves hand meshes as OBJ files or uses CPU-based rendering
to generate videos without requiring GPU/OpenGL context.

Usage:
    python demo_offline.py --video_path video.mp4 --output_format obj
    python demo_offline.py --video_path video.mp4 --output_format images
"""
import argparse
import os
import sys
from pathlib import Path

import cv2
import joblib
import numpy as np
import torch
import trimesh
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from hawor.utils.process import get_mano_faces, run_mano, run_mano_left
from lib.eval_utils.custom_utils import load_slam_cam
from scripts.scripts_test_video.detect_track_video import detect_track_video
from scripts.scripts_test_video.hawor_slam import hawor_slam
from scripts.scripts_test_video.hawor_video import hawor_infiller, hawor_motion_estimation


def save_mesh_sequence_as_obj(vertices, faces, output_dir, prefix="hand"):
    """Save mesh sequence as individual OBJ files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    num_frames = vertices.shape[0]
    for frame_idx in tqdm(range(num_frames), desc=f"Saving {prefix} meshes"):
        mesh = trimesh.Trimesh(
            vertices=vertices[frame_idx],
            faces=faces,
            process=False
        )
        obj_path = output_dir / f"{prefix}_frame_{frame_idx:04d}.obj"
        mesh.export(str(obj_path))

    print(f"✓ Saved {num_frames} {prefix} meshes to {output_dir}")
    return output_dir


def render_mesh_to_image(vertices_left, vertices_right, faces_left, faces_right,
                         frame_idx, img_path, output_size=(1920, 1080)):
    """
    Render meshes to image using CPU-based rendering (pyrender).
    Falls back to simple overlay if pyrender not available.
    """
    try:
        import pyrender

        # Create scene
        scene = pyrender.Scene(ambient_light=[0.3, 0.3, 0.3])

        # Add left hand
        mesh_left = trimesh.Trimesh(vertices=vertices_left[frame_idx], faces=faces_left)
        mesh_left = pyrender.Mesh.from_trimesh(mesh_left, smooth=True)
        scene.add(mesh_left)

        # Add right hand
        mesh_right = trimesh.Trimesh(vertices=vertices_right[frame_idx], faces=faces_right)
        mesh_right = pyrender.Mesh.from_trimesh(mesh_right, smooth=True)
        scene.add(mesh_right)

        # Add camera
        camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0)
        camera_pose = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 2.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        scene.add(camera, pose=camera_pose)

        # Add light
        light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=3.0)
        scene.add(light, pose=camera_pose)

        # Render
        renderer = pyrender.OffscreenRenderer(output_size[0], output_size[1])
        color, depth = renderer.render(scene)
        renderer.delete()

        # Load background image if exists
        if img_path and os.path.exists(img_path):
            bg_img = cv2.imread(img_path)
            bg_img = cv2.resize(bg_img, output_size)
            # Simple alpha blending
            mask = (color.sum(axis=2) > 0).astype(np.uint8) * 255
            mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            result = cv2.addWeighted(bg_img, 0.5, color, 0.5, 0)
        else:
            result = color

        return result

    except ImportError:
        print("⚠️  pyrender not available, falling back to simple visualization")
        # Fallback: just load the background image
        if img_path and os.path.exists(img_path):
            return cv2.imread(img_path)
        else:
            return np.zeros((output_size[1], output_size[0], 3), dtype=np.uint8)


def create_video_from_meshes(vertices_left, vertices_right, faces_left, faces_right,
                             image_paths, output_video_path, fps=30):
    """Create video by rendering meshes frame by frame."""
    output_video_path = Path(output_video_path)
    output_video_path.parent.mkdir(parents=True, exist_ok=True)

    # Get video dimensions from first image
    if image_paths and os.path.exists(image_paths[0]):
        first_img = cv2.imread(image_paths[0])
        height, width = first_img.shape[:2]
    else:
        width, height = 1920, 1080

    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video_path), fourcc, fps, (width, height))

    num_frames = vertices_left.shape[0]
    for frame_idx in tqdm(range(num_frames), desc="Rendering video"):
        img_path = image_paths[frame_idx] if frame_idx < len(image_paths) else None
        frame = render_mesh_to_image(
            vertices_left, vertices_right,
            faces_left, faces_right,
            frame_idx, img_path,
            output_size=(width, height)
        )
        out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    out.release()
    print(f"✓ Video saved to {output_video_path}")
    return output_video_path


def main():
    parser = argparse.ArgumentParser(description="Offline HaWoR visualization")
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, required=True)
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--checkpoint", type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight", type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--output_format", type=str, default='obj',
                       choices=['obj', 'video', 'both'],
                       help='Output format: obj (mesh files), video (rendered), or both')
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
    hand2idx = {"right": 1, "left": 0}
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

    # Output directory
    output_dir = Path(seq_folder) / "offline_vis"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save based on format
    if args.output_format in ['obj', 'both']:
        print("\n=== Saving mesh files ===")
        save_mesh_sequence_as_obj(left_verts, faces_left, output_dir / "left_hand", "left")
        save_mesh_sequence_as_obj(right_verts, faces_right, output_dir / "right_hand", "right")

        # Save a README
        readme_path = output_dir / "README.txt"
        with open(readme_path, 'w') as f:
            f.write("HaWoR Offline Visualization Results\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Video: {args.video_path}\n")
            f.write(f"Frames: {vis_start} to {vis_end}\n")
            f.write(f"Total frames: {vis_end - vis_start + 1}\n\n")
            f.write("Mesh files:\n")
            f.write(f"  - Left hand: left_hand/left_frame_XXXX.obj\n")
            f.write(f"  - Right hand: right_hand/right_frame_XXXX.obj\n\n")
            f.write("You can view these OBJ files in:\n")
            f.write("  - Blender\n")
            f.write("  - MeshLab\n")
            f.write("  - Online viewers (e.g., https://3dviewer.net/)\n")
        print(f"✓ README saved to {readme_path}")

    if args.output_format in ['video', 'both']:
        print("\n=== Rendering video ===")
        video_path = output_dir / "visualization.mp4"
        image_names = [str(f) for f in imgfiles[vis_start:vis_end]]
        create_video_from_meshes(
            left_verts, right_verts,
            faces_left, faces_right,
            image_names, video_path,
            fps=args.fps
        )

    print(f"\n✓ All outputs saved to: {output_dir}")
    print("\nTo view the results:")
    if args.output_format in ['obj', 'both']:
        print(f"  - OBJ files: {output_dir}/left_hand/ and {output_dir}/right_hand/")
        print(f"  - Open any .obj file in Blender, MeshLab, or online viewer")
    if args.output_format in ['video', 'both']:
        print(f"  - Video: {output_dir}/visualization.mp4")


if __name__ == '__main__':
    main()
