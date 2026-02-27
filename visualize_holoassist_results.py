import argparse
import sys
import os

# Use non-interactive backend for matplotlib (no display needed)
import matplotlib
matplotlib.use('Agg')  # Use Anti-Grain Geometry backend (no display)

import torch
import numpy as np
import joblib
import cv2
import matplotlib.pyplot as plt
from matplotlib.animation import FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from glob import glob
from natsort import natsorted
from tqdm import tqdm

# Add current directory to sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "/home/guantianrui/manopth")

# CRITICAL: Use manopth (not smplx) to match Zarr generation
from manopth.manolayer import ManoLayer
from hawor.utils.process import get_mano_faces

# Initialize manopth ManoLayers with Zarr-compatible configuration
_mano_left = None
_mano_right = None

def get_mano_layer_left():
    global _mano_left
    if _mano_left is None:
        _mano_left = ManoLayer(
            mano_root='/home/guantianrui/manopth/mano/models',
            side='left',
            use_pca=False,  # Directly use axis-angle
            flat_hand_mean=True,
            center_idx=0
        )
    return _mano_left

def get_mano_layer_right():
    global _mano_right
    if _mano_right is None:
        _mano_right = ManoLayer(
            mano_root='/home/guantianrui/manopth/mano/models',
            side='right',
            use_pca=False,  # Directly use axis-angle
            flat_hand_mean=True,
            center_idx=0
        )
    return _mano_right

def run_mano_manopth(trans, rot_aa, hand_pose_aa, betas, side='right'):
    """
    Run MANO using manopth (matching build_holoassist_zarr.py logic)
    Args:
        trans: (B, T, 3)
        rot_aa: (B, T, 3)
        hand_pose_aa: (B, T, 45)
        betas: (B, T, 10)
    Returns:
        dict with 'vertices': (B, T, 778, 3)
    """
    layer = get_mano_layer_right() if side == 'right' else get_mano_layer_left()
    
    B, T = trans.shape[0], trans.shape[1]
    
    # Reshape for batch processing
    trans_flat = trans.reshape(B * T, 3)
    rot_aa_flat = rot_aa.reshape(B * T, 3)
    hand_pose_flat = hand_pose_aa.reshape(B * T, 45)
    betas_flat = betas.reshape(B * T, 10)
    
    # Concatenate rotation and hand pose
    full_pose = torch.cat([rot_aa_flat, hand_pose_flat], dim=1)  # (B*T, 48)
    
    # Forward pass
    verts_mm, joints_mm = layer(full_pose, betas_flat)
    
    # Convert mm to meters
    verts = verts_mm / 1000.0
    
    # Add translation
    verts = verts + trans_flat.unsqueeze(1)
    
    # Reshape back
    verts = verts.reshape(B, T, -1, 3)
    
    return {'vertices': verts}

def compute_visibility_masks(pred_valid, max_interp_gap=30):
    """Build per-hand visibility masks from pred_valid, suppressing long interpolated spans."""
    valid = np.asarray(pred_valid).astype(bool)  # (2, T)
    if valid.ndim != 2 or valid.shape[0] != 2:
        raise ValueError(f"pred_valid must have shape (2, T), got {valid.shape}")

    T = valid.shape[1]
    vis = valid.copy()

    for h in range(2):
        # suppress very long synthetic spans between valid detections
        idx = np.where(valid[h])[0]
        if len(idx) < 2:
            continue
        for i in range(len(idx) - 1):
            a, b = idx[i], idx[i + 1]
            gap = b - a - 1
            if gap > max_interp_gap:
                vis[h, a + 1:b] = False

    return vis  # (2, T)


def render_world_view(left_verts, right_verts, left_faces, right_faces, R_c2w, t_c2w, output_path, num_frames, left_vis=None, right_vis=None):
    """在世界坐标系中渲染手部网格"""
    # 设置图形大小
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # 计算所有顶点的范围以设置合适的视角
    # 过滤掉 NaN 和 Inf 值
    all_verts = np.concatenate([left_verts, right_verts], axis=1)
    valid_mask = np.isfinite(all_verts).all(axis=-1)
    if valid_mask.sum() == 0:
        raise ValueError("No valid vertices found (all NaN or Inf)")
    
    valid_verts = all_verts[valid_mask]
    verts_min = valid_verts.min(axis=0)
    verts_max = valid_verts.max(axis=0)
    verts_center = (verts_min + verts_max) / 2
    verts_range = (verts_max - verts_min).max()
    
    # 确保范围不为零或无效
    if not np.isfinite(verts_range) or verts_range < 1e-6:
        verts_range = 1.0  # 默认范围
        verts_center = np.array([0.0, 0.0, 0.0])
    
    # 设置视频写入器
    fps = 30
    writer = FFMpegWriter(fps=fps, metadata=dict(artist='HaWoR'), bitrate=1800)
    
    print(f"Rendering {num_frames} frames...")
    with writer.saving(fig, output_path, dpi=100):
        for frame_idx in tqdm(range(num_frames)):
            ax.clear()
            
            # 绘制左手（紫色）
            if left_vis is None or left_vis[frame_idx]:
                left_v = left_verts[frame_idx]
                # 过滤掉包含 NaN 或 Inf 的面
                valid_left_faces = []
                for face in left_faces[:500]:  # 限制面数以提高速度
                    face_verts = left_v[face]
                    if np.isfinite(face_verts).all():
                        valid_left_faces.append(face_verts)

                if len(valid_left_faces) > 0:
                    left_mesh = Poly3DCollection(valid_left_faces, alpha=0.7, facecolor='purple', edgecolor='darkviolet', linewidths=0.5)
                    ax.add_collection3d(left_mesh)

            # 绘制右手（蓝色）
            if right_vis is None or right_vis[frame_idx]:
                right_v = right_verts[frame_idx]
                valid_right_faces = []
                for face in right_faces[:500]:  # 限制面数以提高速度
                    face_verts = right_v[face]
                    if np.isfinite(face_verts).all():
                        valid_right_faces.append(face_verts)

                if len(valid_right_faces) > 0:
                    right_mesh = Poly3DCollection(valid_right_faces, alpha=0.7, facecolor='blue', edgecolor='darkblue', linewidths=0.5)
                    ax.add_collection3d(right_mesh)
            
            # 绘制相机位置
            if frame_idx < len(t_c2w):
                cam_pos = t_c2w[frame_idx]
                if np.isfinite(cam_pos).all():
                    ax.scatter([cam_pos[0]], [cam_pos[1]], [cam_pos[2]], c='red', s=50, marker='^', label='Camera' if frame_idx == 0 else '')
            
            # 设置坐标轴范围（确保值有效）
            x_min = verts_center[0] - verts_range/2
            x_max = verts_center[0] + verts_range/2
            y_min = verts_center[1] - verts_range/2
            y_max = verts_center[1] + verts_range/2
            z_min = verts_center[2] - verts_range/2
            z_max = verts_center[2] + verts_range/2
            
            # 确保所有值都是有限的
            if np.isfinite([x_min, x_max, y_min, y_max, z_min, z_max]).all():
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                ax.set_zlim(z_min, z_max)
            else:
                # 如果计算出的范围无效，使用默认值
                ax.set_xlim(-1, 1)
                ax.set_ylim(-1, 1)
                ax.set_zlim(-1, 1)
            
            # 设置标签和标题
            ax.set_xlabel('X')
            ax.set_ylabel('Y')
            ax.set_zlabel('Z')
            ax.set_title(f'Frame {frame_idx + 1}/{num_frames}')
            
            # 设置视角
            ax.view_init(elev=20, azim=45 + frame_idx * 0.5)
            
            writer.grab_frame()
    
    plt.close(fig)

def render_camera_view(left_verts, right_verts, left_faces, right_faces, R_w2c, t_w2c, image_names, focal, output_path, num_frames, left_vis=None, right_vis=None, show_hand_ids=True):
    """在相机坐标系中渲染，将手部网格投影到原始图像上，检查贴合度"""
    # 读取第一张图像获取尺寸
    first_img = cv2.imread(image_names[0])
    if first_img is None:
        raise FileNotFoundError(f"Cannot read image: {image_names[0]}")
    
    height, width = first_img.shape[:2]
    
    # 设置视频写入器
    fps = 30
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # 相机内参
    K = np.array([
        [focal, 0, width / 2],
        [0, focal, height / 2],
        [0, 0, 1]
    ], dtype=np.float32)
    
    print(f"Rendering {num_frames} frames with hand projection onto images...")
    for frame_idx in tqdm(range(num_frames)):
        # 读取原始图像
        if frame_idx < len(image_names):
            img = cv2.imread(image_names[frame_idx])
            if img is None:
                img = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            img = np.zeros((height, width, 3), dtype=np.uint8)
        
        # 创建叠加图像（半透明手部网格）
        overlay = img.copy()
        
        # 将手部顶点投影到图像平面
        if frame_idx < len(R_w2c):
            R = R_w2c[frame_idx]
            t = t_w2c[frame_idx]
            
            # 投影左手
            left_label_pos = None
            if left_vis is None or left_vis[frame_idx]:
                left_v = left_verts[frame_idx]  # (N, 3)
                left_v_cam = (R @ left_v.T).T + t  # 转换到相机坐标系
                # 只投影在相机前方的点
                valid_left = left_v_cam[:, 2] > 0.01  # 避免z=0的情况

                if valid_left.sum() > 0:
                    # 投影到2D
                    left_v_2d = np.zeros((len(left_v), 2), dtype=np.float32)
                    left_v_cam_valid = left_v_cam[valid_left]
                    left_v_2d[valid_left] = (K @ left_v_cam_valid.T).T[:, :2] / left_v_cam_valid[:, 2:3]
                    # 编号锚点：可见顶点中心
                    left_label_pos = np.median(left_v_2d[valid_left], axis=0)

                    # 绘制左手网格（使用半透明填充）
                    for face in left_faces:
                        if np.all(valid_left[face]):
                            pts = left_v_2d[face].astype(np.int32)
                            # 检查点是否在图像范围内（允许一些边界外的点）
                            if np.any((pts[:, 0] >= 0) & (pts[:, 0] < width) & 
                                      (pts[:, 1] >= 0) & (pts[:, 1] < height)):
                                # 只绘制在图像内的部分
                                pts_clipped = np.clip(pts, [0, 0], [width-1, height-1])
                                cv2.fillPoly(overlay, [pts_clipped], (255, 0, 255), lineType=cv2.LINE_AA)  # 紫色 (BGR)
                                # 绘制边缘
                                cv2.polylines(overlay, [pts_clipped], True, (200, 0, 200), 1, lineType=cv2.LINE_AA)

            # 投影右手
            right_label_pos = None
            if right_vis is None or right_vis[frame_idx]:
                right_v = right_verts[frame_idx]  # (N, 3)
                right_v_cam = (R @ right_v.T).T + t
                # 只投影在相机前方的点
                valid_right = right_v_cam[:, 2] > 0.01

                if valid_right.sum() > 0:
                    # 投影到2D
                    right_v_2d = np.zeros((len(right_v), 2), dtype=np.float32)
                    right_v_cam_valid = right_v_cam[valid_right]
                    right_v_2d[valid_right] = (K @ right_v_cam_valid.T).T[:, :2] / right_v_cam_valid[:, 2:3]
                    # 编号锚点：可见顶点中心
                    right_label_pos = np.median(right_v_2d[valid_right], axis=0)

                    # 绘制右手网格
                    for face in right_faces:
                        if np.all(valid_right[face]):
                            pts = right_v_2d[face].astype(np.int32)
                            if np.any((pts[:, 0] >= 0) & (pts[:, 0] < width) & 
                                      (pts[:, 1] >= 0) & (pts[:, 1] < height)):
                                pts_clipped = np.clip(pts, [0, 0], [width-1, height-1])
                                cv2.fillPoly(overlay, [pts_clipped], (255, 0, 0), lineType=cv2.LINE_AA)  # 蓝色 (BGR)
                                # 绘制边缘
                                cv2.polylines(overlay, [pts_clipped], True, (200, 0, 0), 1, lineType=cv2.LINE_AA)
        
        # 将叠加层与原始图像混合（半透明效果）
        alpha = 0.6  # 透明度
        result = cv2.addWeighted(img, 1 - alpha, overlay, alpha, 0)
        
        # 添加帧号信息
        cv2.putText(result, f'Frame {frame_idx + 1}/{num_frames}', (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(result, 'Left: Purple, Right: Blue', (10, height - 20), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # 手部编号（Left=0, Right=1）
        if show_hand_ids:
            if left_label_pos is not None and np.isfinite(left_label_pos).all():
                lx, ly = int(np.clip(left_label_pos[0], 0, width - 1)), int(np.clip(left_label_pos[1], 0, height - 1))
                cv2.putText(result, 'L0', (lx, ly), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 120, 255), 2)
            if right_label_pos is not None and np.isfinite(right_label_pos).all():
                rx, ry = int(np.clip(right_label_pos[0], 0, width - 1)), int(np.clip(right_label_pos[1], 0, height - 1))
                cv2.putText(result, 'R1', (rx, ry), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 120, 120), 2)

        out.write(result)
    
    out.release()
    print(f"Video saved to: {output_path}")

def visualize_results(result_file, output_dir=None, vis_mode='world', vis_start=None, vis_end=None, max_interp_gap=30, show_hand_ids=True):
    """
    从保存的结果文件可视化手部姿态
    
    Args:
        result_file: hawor_results.pkl 文件路径
        output_dir: 可视化输出目录，如果为None则使用结果文件所在目录
        vis_mode: 'world' 或 'cam'
        vis_start: 起始帧索引，如果为None则从0开始
        vis_end: 结束帧索引，如果为None则到最后一帧
    """
    # 加载结果
    print(f"Loading results from {result_file}...")
    results = joblib.load(result_file)
    
    video_id = results['video_id']
    pred_trans = torch.from_numpy(results['pred_trans'])  # (2, T, 3)
    pred_rot = torch.from_numpy(results['pred_rot'])  # (2, T, 3)
    pred_hand_pose = torch.from_numpy(results['pred_hand_pose'])  # (2, T, 45)
    pred_betas = torch.from_numpy(results['pred_betas'])  # (2, T, 10)
    pred_valid = results['pred_valid']  # (2, T)
    vis_masks = compute_visibility_masks(pred_valid, max_interp_gap=max_interp_gap)
    
    R_w2c_sla_all = torch.from_numpy(results['slam_R_w2c'])
    t_w2c_sla_all = torch.from_numpy(results['slam_t_w2c'])
    
    # Handle missing c2w keys (backward compatibility)
    if 'slam_R_c2w' in results:
        R_c2w_sla_all = torch.from_numpy(results['slam_R_c2w'])
        t_c2w_sla_all = torch.from_numpy(results['slam_t_c2w'])
    else:
        print("Computing R_c2w and t_c2w from w2c (missing in results)...")
        R_c2w_sla_all = R_w2c_sla_all.transpose(-1, -2)
        t_c2w_sla_all = -torch.einsum("bij,bj->bi", R_c2w_sla_all, t_w2c_sla_all)

    img_focal = results['img_focal']
    
    # 确定可视化范围
    num_frames = pred_trans.shape[1]
    if vis_start is None:
        vis_start = 0
    if vis_end is None:
        vis_end = num_frames - 1
    
    vis_start = max(0, min(vis_start, num_frames - 1))
    vis_end = max(vis_start, min(vis_end, num_frames - 1))
    
    print(f"Visualizing frames {vis_start} to {vis_end} (total {num_frames} frames)")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(result_file)
    vis_output_dir = os.path.join(output_dir, f"vis_{vis_start}_{vis_end}")
    os.makedirs(vis_output_dir, exist_ok=True)
    
    # 查找图像文件
    # 尝试在结果目录下查找 extracted_images
    result_dir = os.path.dirname(result_file)
    img_folder = os.path.join(result_dir, 'extracted_images')
    
    if not os.path.exists(img_folder):
        # 尝试在上级目录查找
        parent_dir = os.path.dirname(result_dir)
        img_folder = os.path.join(parent_dir, video_id, 'extracted_images')
    
    if not os.path.exists(img_folder):
        print(f"Warning: Could not find extracted_images folder. Looking in {result_dir}...")
        # 尝试递归查找
        for root, dirs, files in os.walk(result_dir):
            if 'extracted_images' in dirs:
                img_folder = os.path.join(root, 'extracted_images')
                break
    
    if not os.path.exists(img_folder):
        raise FileNotFoundError(f"Could not find extracted_images folder for {video_id}")
    
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
    if len(imgfiles) == 0:
        imgfiles = natsorted(glob(f'{img_folder}/*.png'))
    
    if len(imgfiles) == 0:
        raise FileNotFoundError(f"No images found in {img_folder}")
    
    print(f"Found {len(imgfiles)} images in {img_folder}")
    image_names = imgfiles[vis_start:vis_end+1]
    
    # 获取 MANO faces
    faces = get_mano_faces()
    faces_new = np.array([[92, 38, 234],
            [234, 38, 239],
            [38, 122, 239],
            [239, 122, 279],
            [122, 118, 279],
            [279, 118, 215],
            [118, 117, 215],
            [215, 117, 214],
            [117, 119, 214],
            [214, 119, 121],
            [119, 120, 121],
            [121, 120, 78],
            [120, 108, 78],
            [78, 108, 79]])
    faces_right = np.concatenate([faces, faces_new], axis=0)
    
    # 生成右手网格 (using manopth to match Zarr)
    hand2idx = {"right": 1, "left": 0}
    hand_idx = hand2idx["right"]
    pred_glob_r = run_mano_manopth(
        pred_trans[hand_idx:hand_idx+1, vis_start:vis_end+1],
        pred_rot[hand_idx:hand_idx+1, vis_start:vis_end+1],
        pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end+1],
        betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end+1],
        side='right'
    )
    right_verts = pred_glob_r['vertices'][0]  # (T, N, 3)
    right_dict = {
        'vertices': right_verts.unsqueeze(0),  # (1, T, N, 3)
        'faces': faces_right,
    }
    
    # 生成左手网格 (using manopth to match Zarr)
    faces_left = faces_right[:, [0, 2, 1]]
    hand_idx = hand2idx["left"]
    pred_glob_l = run_mano_manopth(
        pred_trans[hand_idx:hand_idx+1, vis_start:vis_end+1],
        pred_rot[hand_idx:hand_idx+1, vis_start:vis_end+1],
        pred_hand_pose[hand_idx:hand_idx+1, vis_start:vis_end+1],
        betas=pred_betas[hand_idx:hand_idx+1, vis_start:vis_end+1],
        side='left'
    )
    left_verts = pred_glob_l['vertices'][0]  # (T, N, 3)
    left_dict = {
        'vertices': left_verts.unsqueeze(0),  # (1, T, N, 3)
        'faces': faces_left,
    }
    
    # DISABLED: Coordinate transformation (R_x flip)
    # We now use HaWoR coordinates directly, matching Zarr definition
    # R_x = torch.tensor([[1,  0,  0],
    #                     [0, -1,  0],
    #                     [0,  0, -1]]).float()
    # R_c2w_sla_all = torch.einsum('ij,njk->nik', R_x, R_c2w_sla_all)
    # t_c2w_sla_all = torch.einsum('ij,nj->ni', R_x, t_c2w_sla_all)
    # R_w2c_sla_all = R_c2w_sla_all.transpose(-1, -2)
    # t_w2c_sla_all = -torch.einsum("bij,bj->bi", R_w2c_sla_all, t_c2w_sla_all)
    # 
    # left_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, left_dict['vertices'].cpu())
    # right_dict['vertices'] = torch.einsum('ij,btnj->btni', R_x, right_dict['vertices'].cpu())
    
    # Use HaWoR coordinates directly (no R_x transformation)
    # W2C is already computed correctly from C2W above
    
    # 将可见性 mask 对齐到当前可视化区间
    left_vis = vis_masks[0, vis_start:vis_end+1]
    right_vis = vis_masks[1, vis_start:vis_end+1]

    # 可视化 - 使用 matplotlib 和 OpenCV，不依赖 OpenGL
    print(f"Rendering visualization in {vis_mode} mode...")
    
    # 转换为 numpy
    left_verts_np = left_dict['vertices'][0].cpu().numpy()  # (T, N, 3)
    right_verts_np = right_dict['vertices'][0].cpu().numpy()  # (T, N, 3)
    left_faces = left_dict['faces']
    right_faces = faces_right
    
    num_frames_vis = vis_end - vis_start + 1
    
    # 创建输出视频路径
    output_video = os.path.join(vis_output_dir, 'visualization.mp4')
    
    if vis_mode == 'world':
        # 世界坐标系可视化
        render_world_view(left_verts_np, right_verts_np, left_faces, right_faces,
                        R_c2w_sla_all[vis_start:vis_end+1].cpu().numpy(),
                        t_c2w_sla_all[vis_start:vis_end+1].cpu().numpy(),
                        output_video, num_frames_vis,
                        left_vis=left_vis, right_vis=right_vis)
    elif vis_mode == 'cam':
        # 相机坐标系可视化
        render_camera_view(left_verts_np, right_verts_np, left_faces, right_faces,
                          R_w2c_sla_all[vis_start:vis_end+1].cpu().numpy(),
                          t_w2c_sla_all[vis_start:vis_end+1].cpu().numpy(),
                          image_names, img_focal, output_video, num_frames_vis,
                          left_vis=left_vis, right_vis=right_vis,
                          show_hand_ids=show_hand_ids)
    else:
        raise ValueError(f"Unknown vis_mode: {vis_mode}. Must be 'world' or 'cam'")
    
    print(f"Visualization saved to: {output_video}")
    print("Visualization complete!")

def main():
    parser = argparse.ArgumentParser(description='Visualize HaWoR results and save video (projects hands onto images)')
    parser.add_argument("--result_file", type=str, required=True,
                       help="Path to hawor_results.pkl file")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Output directory for visualization (default: same as result file)")
    parser.add_argument("--vis_mode", type=str, default='cam', choices=['world', 'cam'],
                       help="Visualization mode: 'cam' (project onto images, default) or 'world' (3D view)")
    parser.add_argument("--start_frame", type=int, default=None,
                       help="Start frame index (default: 0)")
    parser.add_argument("--end_frame", type=int, default=None,
                       help="End frame index (default: last frame)")
    parser.add_argument("--max_interp_gap", type=int, default=30,
                       help="Maximum interpolated gap to keep visible; longer gaps are hidden in visualization")
    parser.add_argument("--show_hand_ids", action='store_true',
                       help="Overlay hand IDs in camera view (L0/R1)")
    
    args = parser.parse_args()
    
    # Ensure we are in the project root
    project_root = os.path.dirname(os.path.abspath(__file__))
    if os.getcwd() != project_root:
        print(f"Changing working directory to {project_root}")
        os.chdir(project_root)
    
    visualize_results(
        args.result_file,
        args.output_dir,
        args.vis_mode,
        args.start_frame,
        args.end_frame,
        args.max_interp_gap,
        args.show_hand_ids
    )

if __name__ == '__main__':
    main()

