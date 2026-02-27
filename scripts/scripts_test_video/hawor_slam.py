import math
import sys
import os

from natsort import natsorted

sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import argparse
from tqdm import tqdm
import numpy as np
import torch
import cv2
from PIL import Image
from glob import glob
from pycocotools import mask as masktool
from lib.pipeline.masked_droid_slam import *
from lib.pipeline.est_scale import *
from hawor.utils.process import block_print, enable_print

sys.path.insert(0, os.path.dirname(__file__) + '/../../thirdparty/Metric3D')
from metric import Metric3D


def get_all_mp4_files(folder_path):
    # Ensure the folder path is absolute
    folder_path = os.path.abspath(folder_path)
    
    # Recursively search for all .mp4 files in the folder and its subfolders
    mp4_files = glob(os.path.join(folder_path, '**', '*.mp4'), recursive=True)
    
    return mp4_files

def split_list_by_interval(lst, interval=1000):
    start_indices = []
    end_indices = []
    split_lists = []
    
    for i in range(0, len(lst), interval):
        start_indices.append(i)
        end_indices.append(min(i + interval, len(lst)))
        split_lists.append(lst[i:i + interval])
    
    return start_indices, end_indices, split_lists

_metric_model_cache = None

def get_metric_model(weight_path):
    global _metric_model_cache
    if _metric_model_cache is None:
        block_print()
        _metric_model_cache = Metric3D(weight_path)
        enable_print()
    return _metric_model_cache

def hawor_slam(args, start_idx, end_idx, seq_folder=None):
    # File and folders
    # 如果提供了 seq_folder，使用它；否则从 video_path 推导（向后兼容）
    if seq_folder is None:
        if hasattr(args, 'seq_folder') and args.seq_folder:
            seq_folder = args.seq_folder
        else:
            file = args.video_path
            video_root = os.path.dirname(file)
            video = os.path.basename(file).split('.')[0]
            seq_folder = os.path.join(video_root, video)
    
    os.makedirs(seq_folder, exist_ok=True)
    video_folder = seq_folder
    img_folder = f'{video_folder}/extracted_images'
    imgfiles = natsorted(glob(f'{img_folder}/*.jpg'))
    
    # 检查图像文件是否存在
    if len(imgfiles) == 0:
        raise ValueError(f"No images found in {img_folder}. Please extract frames first.")

    first_img = cv2.imread(imgfiles[0])
    height, width, _ = first_img.shape

    print(f'Running slam on {video_folder} ...')

    ##### Run SLAM #####
    # Use Masking
    masks_path = f'{video_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy'
    if not os.path.exists(masks_path):
        raise ValueError(f"Model masks not found at {masks_path}. Please run Stage 1 (motion estimation) first.")
    masks = np.load(masks_path, allow_pickle=True)
    masks = torch.from_numpy(masks)
    print(masks.shape)

    # Camera calibration (intrinsics) for SLAM
    focal = args.img_focal
    if focal is None:
        try:
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'r') as file:
                focal = file.read()
                focal = float(focal)
        except:
            
            print('No focal length provided')
            focal = 600
            with open(os.path.join(seq_folder, 'est_focal.txt'), 'w') as file:
                file.write(str(focal))
    calib = np.array(est_calib(imgfiles)) # [focal, focal, cx, cy]
    center = calib[2:]        
    calib[:2] = focal

    # Droid-slam with masking
    droid, traj = run_slam(imgfiles, masks=masks, calib=calib)
    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    print('DBA errors:', droid.backend.errors)

    del droid
    torch.cuda.empty_cache()

    # Estimate scale  
    block_print()  
    metric = get_metric_model('thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth')
    enable_print() 
    min_threshold = 0.4
    max_threshold = 0.7

    # ===== 性能优化：采样策略 =====
    # 对于长视频，不要对所有 keyframe 都进行深度估计
    # 采样策略：基于比例的智能采样，根据视频长度自适应调整
    n_keyframes = len(tstamp)
    
    # 可配置的采样参数（可以通过 args 传入）
    max_samples = getattr(args, 'metric3d_max_samples', 200)  # 最多采样帧数（默认 200，适合长视频）
    min_samples = getattr(args, 'metric3d_min_samples', 10)  # 至少采样帧数（默认 10）
    sample_ratio = getattr(args, 'metric3d_sample_ratio', 0.15)  # 采样比例（默认 15%，即采样 15% 的 keyframe）
    
    # 智能采样策略：基于比例 + 上下限
    if n_keyframes <= min_samples:
        # 如果 keyframe 很少，全部采样
        sample_indices = list(range(n_keyframes))
        print(f'Metric3D sampling: All {n_keyframes} keyframes (video too short)')
    else:
        # 基于比例计算目标采样数
        target_samples = int(n_keyframes * sample_ratio)
        # 应用上下限
        target_samples = max(min_samples, min(target_samples, max_samples))
        
        # 计算采样间隔
        sample_interval = max(1, n_keyframes // target_samples)
        
        # 生成采样索引
        sample_indices = list(range(0, n_keyframes, sample_interval))
        
        # 确保包含第一帧和最后一帧
        if sample_indices[0] != 0:
            sample_indices.insert(0, 0)
        if sample_indices[-1] != n_keyframes - 1:
            sample_indices.append(n_keyframes - 1)
        
        # 如果采样过多（可能因为包含首尾帧），均匀下采样到目标数量
        if len(sample_indices) > target_samples:
            # 保留首尾帧，中间均匀采样
            if target_samples >= 3:
                # 保留首尾
                keep_indices = [0, n_keyframes - 1]
                # 中间部分均匀采样
                middle_indices = [i for i in sample_indices if i not in keep_indices]
                if len(middle_indices) > 0:
                    step = len(middle_indices) / (target_samples - 2)
                    selected_middle = [middle_indices[int(i * step)] for i in range(target_samples - 2)]
                    sample_indices = sorted(keep_indices + selected_middle)
                else:
                    sample_indices = keep_indices
            else:
                # 如果目标数量太少，只保留首尾
                sample_indices = [0, n_keyframes - 1][:target_samples]
        
        print(f'Metric3D sampling: {len(sample_indices)}/{n_keyframes} keyframes ({len(sample_indices)/n_keyframes*100:.1f}%, speedup: {n_keyframes/len(sample_indices):.1f}x)')

    print('Predicting Metric Depth (sampled keyframes only) ...')
    pred_depths_dict = {}  # 只存储采样帧的深度
    H, W = get_dimention(imgfiles)
    for idx in tqdm(sample_indices, desc="Metric3D depth"):
        t = tstamp[idx]
        pred_depth = metric(imgfiles[t], calib)
        pred_depth = cv2.resize(pred_depth, (W, H))
        pred_depths_dict[idx] = pred_depth

    ##### Estimate Metric Scale (only on sampled frames) #####
    print('Estimating Metric Scale (on sampled frames) ...')
    scales_ = []
    for idx in tqdm(sample_indices, desc="Scale estimation"):
        t = tstamp[idx]
        disp = disps[idx]
        pred_depth = pred_depths_dict[idx]
        slam_depth = 1/disp
        
        # Estimate scene scale
        msk = masks[t].numpy().astype(np.uint8)
        scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=min_threshold, far_thresh=max_threshold)  
        while math.isnan(scale):
            min_threshold -= 0.1
            max_threshold += 0.1
            scale = est_scale_hybrid(slam_depth, pred_depth, sigma=0.5, msk=msk, near_thresh=min_threshold, far_thresh=max_threshold)                    
        scales_.append(scale)

    median_s = np.median(scales_)
    print(f"estimated scale: {median_s} (from {len(scales_)} sampled frames)")

    # Save results
    os.makedirs(f"{seq_folder}/SLAM", exist_ok=True)
    save_path = f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz'
    np.savez(save_path, 
            tstamp=tstamp, disps=disps, traj=traj, 
            img_focal=focal, img_center=calib[-2:],
            scale=median_s)