import math
import sys
import os

sys.path.insert(0, os.path.dirname(__file__) + '/../..')

import argparse
from glob import glob
from tqdm import tqdm
import numpy as np
import torch
import cv2
from pycocotools import mask as masktool
from lib.pipeline.masked_droid_slam import *
from lib.pipeline.est_scale import *
from lib.pipeline.frame_source import build_frame_source_auto
from hawor.utils.process import block_print, enable_print

sys.path.insert(0, os.path.dirname(__file__) + '/../../thirdparty/Metric3D')
from metric import Metric3D

# Check if we should suppress verbose output
QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"

def vprint(*args, **kwargs):
    """Print only if not in quiet mode."""
    if not QUIET_MODE:
        print(*args, **kwargs)


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

def build_metric3d_runner(weight_path='thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth'):
    block_print()
    metric = Metric3D(weight_path)
    enable_print()
    return metric


def hawor_slam(args, start_idx, end_idx, metric_runner=None, metric3d_batch_size=8):
    # File and folders
    file = args.video_path
    video_root = os.path.dirname(file)
    video = os.path.basename(file).split('.')[0]
    seq_folder = os.path.join(video_root, video)
    os.makedirs(seq_folder, exist_ok=True)
    video_folder = os.path.join(video_root, video)

    frame_backend = getattr(args, 'frame_backend', 'decord')
    frame_source, _ = build_frame_source_auto(file, backend=frame_backend)

    first_img = frame_source.get_frame(0, rgb=False)
    height, width, _ = first_img.shape
    
    vprint(f'Running slam on {video_folder} ...')

    ##### Run SLAM #####
    # Use Masking
    masks = np.load(f'{video_folder}/tracks_{start_idx}_{end_idx}/model_masks.npy', allow_pickle=True)
    masks = torch.from_numpy(masks)
    vprint(masks.shape)

    # Camera calibration (intrinsics) for SLAM
    focal = args.img_focal
    if focal is None:
        try:
            with open(os.path.join(video_folder, 'est_focal.txt'), 'r') as file:
                focal = file.read()
                focal = float(focal)
        except:
            
            vprint('No focal length provided')
            focal = 600
            with open(os.path.join(video_folder, 'est_focal.txt'), 'w') as file:
                file.write(str(focal))
    calib = np.array(est_calib(frame_source)) # [focal, focal, cx, cy]
    center = calib[2:]        
    calib[:2] = focal
    
    # Droid-slam with masking
    droid, traj = run_slam(frame_source, masks=masks, calib=calib)
    n = droid.video.counter.value
    tstamp = droid.video.tstamp.cpu().int().numpy()[:n]
    disps = droid.video.disps_up.cpu().numpy()[:n]
    vprint('DBA errors:', droid.backend.errors)

    del droid
    torch.cuda.empty_cache()

    # Estimate scale
    metric = metric_runner or build_metric3d_runner()
    min_threshold = 0.4
    max_threshold = 0.7

    vprint('Predicting Metric Depth ...')
    pred_depths = []
    H, W = get_dimention(frame_source)

    # Batch processing for Metric3D
    num_frames = len(tstamp)
    for batch_start in tqdm(range(0, num_frames, metric3d_batch_size), desc="Metric3D batches", disable=QUIET_MODE):
        batch_end = min(batch_start + metric3d_batch_size, num_frames)
        batch_indices = tstamp[batch_start:batch_end]

        # Load batch of frames (list comprehension allows backend optimization)
        batch_frames = [frame_source.get_frame(int(t), rgb=True) for t in batch_indices]

        # Batch inference
        batch_depths = metric.batch_inference(batch_frames, calib)

        # Resize and append results
        for pred_depth in batch_depths:
            pred_depth = cv2.resize(pred_depth, (W, H))
            pred_depths.append(pred_depth)

    ##### Estimate Metric Scale #####
    vprint('Estimating Metric Scale ...')
    scales_ = []
    n = len(tstamp)   # for each keyframe
    for i in tqdm(range(n), disable=QUIET_MODE):
        t = tstamp[i]
        disp = disps[i]
        pred_depth = pred_depths[i]
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
    vprint(f"estimated scale: {median_s}")

    # Save results
    os.makedirs(f"{seq_folder}/SLAM", exist_ok=True)
    save_path = f'{seq_folder}/SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz'
    np.savez(save_path,
            tstamp=tstamp, disps=disps, traj=traj,
            img_focal=focal, img_center=calib[-2:],
            scale=median_s)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--frame_backend", type=str, default='decord', choices=['decord', 'opencv'])
    args = parser.parse_args()

    # Need detect_track first to get track indices
    from scripts.scripts_test_video.detect_track_video import detect_track_video
    start_idx, end_idx, _, _ = detect_track_video(args)
    hawor_slam(args, start_idx, end_idx)

