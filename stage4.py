import argparse
import sys
import os
import numpy as np
import joblib
from tqdm import tqdm
import traceback
from multiprocessing import Process, set_start_method
from glob import glob
from natsort import natsorted
import logging
from datetime import datetime
import json
from pathlib import Path

# ==========================================
# 0. 基础配置
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

# [配置] 原始数据的根目录，用于查找 intrinsics.json
RAW_DATASET_ROOT = "/share_data/guantianrui/datasets/Egocentric-10K"

# ==========================================
# 1. 增强日志系统
# ==========================================
_log_file_path = None

def setup_logging(output_dir):
    global _log_file_path
    log_dir = os.path.join(output_dir, 'logs')
    os.makedirs(log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"pipeline_hawor_{timestamp}.log")
    _log_file_path = log_file

    root_logger = logging.getLogger()
    for h in root_logger.handlers[:]: root_logger.removeHandler(h)

    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))

    root_logger.setLevel(logging.INFO)
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    return log_file

def setup_subprocess_logging(log_file_path):
    if log_file_path and os.path.exists(os.path.dirname(log_file_path)):
        root_logger = logging.getLogger()
        for h in root_logger.handlers[:]: root_logger.removeHandler(h)
        file_handler = logging.FileHandler(log_file_path, mode='a', encoding='utf-8')
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', '%Y-%m-%d %H:%M:%S'))
        root_logger.setLevel(logging.INFO)
        root_logger.addHandler(file_handler)
        file_handler.stream.flush()

def log_msg(msg, level="info"):
    if level == "info": logging.info(msg)
    elif level == "error": logging.error(msg)
    elif level == "warn": logging.warning(msg)
    for handler in logging.getLogger().handlers:
        if isinstance(handler, logging.FileHandler):
            handler.stream.flush()

# ==========================================
# 2. 辅助函数
# ==========================================

def get_video_id(video_path):
    return os.path.splitext(os.path.basename(video_path))[0]

def get_focal_length_from_intrinsics(video_path):
    try:
        v_path = Path(video_path)
        worker_name = v_path.parent.parent.name
        factory_name = v_path.parent.parent.parent.name
        json_path = Path(RAW_DATASET_ROOT) / factory_name / "workers" / worker_name / "intrinsics.json"
        if not json_path.exists():
            return None
        with open(json_path, 'r') as f:
            data = json.load(f)
            return float(data['fx'])
    except Exception as e:
        log_msg(f"Error reading intrinsics for {video_path}: {e}", level="warn")
        return None

def clear_gpu_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except:
        pass

# ==========================================
# 3. GPU Worker 架构
# ==========================================

def process_single_video(worker_id, vi, total, video_path, output_root, args_dict, log_file_path):
    """处理单个视频：抽帧 → 检测 → 运动估计。模型已由 worker 预加载到全局缓存。"""
    import copy
    import time
    from scripts.scripts_test_video.detect_track_video import extract_frames
    from lib.pipeline.tools import detect_track
    from scripts.scripts_test_video.hawor_video import hawor_motion_estimation

    start_time = time.time()
    try:
        args = argparse.Namespace(**copy.deepcopy(args_dict))
        video_id = get_video_id(video_path)
        video_out_dir = os.path.abspath(os.path.join(output_root, video_id))
        os.makedirs(video_out_dir, exist_ok=True)

        # 读取内参
        focal_length = get_focal_length_from_intrinsics(video_path)
        if focal_length:
            args.img_focal = focal_length
            log_msg(f"[W{worker_id}][{vi+1}/{total}][{video_id}] intrinsics fx={focal_length:.2f}")
        else:
            args.img_focal = None

        log_msg(f"[W{worker_id}][{vi+1}/{total}][{video_id}] Stage 1 Started")

        # 1. 抽帧
        img_folder = os.path.join(video_out_dir, 'extracted_images')
        os.makedirs(img_folder, exist_ok=True)
        imgfiles = natsorted(glob(os.path.join(img_folder, '*.jpg')))

        if len(imgfiles) == 0:
            log_msg(f"[{video_id}] Extracting frames...")
            extract_start = time.time()
            extract_frames(video_path, img_folder)
            imgfiles = natsorted(glob(os.path.join(img_folder, '*.jpg')))
            log_msg(f"[{video_id}] Extracted {len(imgfiles)} frames in {time.time()-extract_start:.1f}s")

        if len(imgfiles) == 0:
            log_msg(f"[{video_id}] WARNING: No frames extracted, skipping...")
            return

        start_idx, end_idx = 0, len(imgfiles)

        # 2. 检测（模型已在全局缓存中，detect_track 会自动复用）
        track_dir = os.path.join(video_out_dir, f"tracks_{start_idx}_{end_idx}")
        track_file = os.path.join(track_dir, 'model_tracks.npy')

        if not os.path.exists(track_file):
            log_msg(f"[{video_id}] Running detection & tracking...")
            detect_start = time.time()
            os.makedirs(track_dir, exist_ok=True)
            clear_gpu_memory()

            max_retries = 2
            tracks = None
            for retry in range(max_retries):
                try:
                    _, tracks = detect_track(imgfiles, thresh=0.5)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and retry < max_retries - 1:
                        log_msg(f"[{video_id}] OOM in detection, retrying...")
                        clear_gpu_memory()
                        time.sleep(2)
                    else:
                        raise

            np.save(track_file, tracks if tracks is not None else {})
            np.save(os.path.join(track_dir, 'model_boxes.npy'), np.array([]))

            try:
                track_count = len(tracks) if tracks is not None else 0
            except TypeError:
                track_count = "unknown"

            log_msg(f"[{video_id}] Detection done: {track_count} tracks in {time.time()-detect_start:.1f}s")
            clear_gpu_memory()

        # 3. 运动估计（HAWOR 模型已在全局缓存中）
        motion_cache = os.path.join(video_out_dir, f"motion_estimation_{start_idx}_{end_idx}.pkl")
        if not os.path.exists(motion_cache):
            log_msg(f"[{video_id}] Running motion estimation...")
            motion_start = time.time()
            args.video_path = video_path

            clear_gpu_memory()
            max_retries = 2
            for retry in range(max_retries):
                try:
                    frame_chunks_all, est_focal = hawor_motion_estimation(args, start_idx, end_idx, video_out_dir)
                    break
                except RuntimeError as e:
                    if "out of memory" in str(e).lower() and retry < max_retries - 1:
                        log_msg(f"[{video_id}] OOM in motion est, retrying...")
                        clear_gpu_memory()
                        time.sleep(2)
                    else:
                        raise

            final_focal = args.img_focal if args.img_focal else est_focal
            joblib.dump({'frame_chunks_all': frame_chunks_all, 'img_focal': final_focal}, motion_cache)
            log_msg(f"[{video_id}] Motion estimation done in {time.time()-motion_start:.1f}s (focal={final_focal:.1f})")
            clear_gpu_memory()

        total_time = time.time() - start_time
        log_msg(f"[W{worker_id}][{vi+1}/{total}][{video_id}] Stage 1 Completed in {total_time:.1f}s")

    except Exception as e:
        log_msg(f"[{get_video_id(video_path)}] ERROR Stage 1: {e}", level="error")
        log_msg(traceback.format_exc(), level="error")


def gpu_worker(worker_id, gpu_id_str, video_list, output_root, args_dict, log_file_path):
    """
    GPU Worker 进程：
    1. 绑定到指定 GPU
    2. 一次性加载所有模型（YOLO + HAWOR）
    3. 循环处理分配的所有视频，复用模型
    """
    setup_subprocess_logging(log_file_path)

    # 绑定 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id_str

    import torch
    import time
    import random

    time.sleep(random.uniform(0.1, 0.5))
    if torch.cuda.is_available():
        torch.cuda.set_device(0)
        try:
            _ = torch.zeros(1, device='cuda')
            torch.cuda.synchronize()
        except Exception:
            time.sleep(0.5)
            torch.cuda.set_device(0)
            _ = torch.zeros(1, device='cuda')
            torch.cuda.synchronize()
        torch.cuda.empty_cache()

    try:
        import cv2
        cv2.setNumThreads(0)
    except:
        pass

    # ---- 一次性加载所有模型 ----
    log_msg(f"[W{worker_id}] GPU {gpu_id_str}: Loading models...")
    load_start = time.time()

    # 1) YOLO 检测模型 → 设置到 tools 模块的全局缓存
    from ultralytics import YOLO
    from lib.pipeline import tools as tools_module
    weights_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'weights', 'external', 'detector.pt')
    if not os.path.exists(weights_path):
        weights_path = './weights/external/detector.pt'
    yolo_model = YOLO(weights_path)
    if torch.cuda.is_available():
        yolo_model.to('cuda')
    tools_module._hand_det_model_cache = yolo_model

    # 2) HAWOR 运动估计模型 → 设置到 hawor_video 模块的全局缓存
    import scripts.scripts_test_video.hawor_video as hawor_video_module
    from scripts.scripts_test_video.hawor_video import load_hawor
    model, model_cfg = load_hawor(args_dict['checkpoint'])
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()
    hawor_video_module._hawor_model_cache = model
    hawor_video_module._hawor_model_cfg_cache = model_cfg

    log_msg(f"[W{worker_id}] GPU {gpu_id_str}: Models loaded in {time.time()-load_start:.1f}s, processing {len(video_list)} videos")

    # ---- 循环处理视频 ----
    for vi, video_path in enumerate(video_list):
        process_single_video(worker_id, vi, len(video_list), video_path, output_root, args_dict, log_file_path)

# ==========================================
# 4. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='/share_data/lvjianan/datasets/BuildAI-processed')
    parser.add_argument("--output_dir", type=str, default='/share_data/lvjianan/HaWoR/output/buildai_results')
    parser.add_argument("--checkpoint", type=str, default='./weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight", type=str, default='./weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of worker processes per GPU")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--n", type=int, default=-1)
    parser.add_argument("--tmp_dir", type=str, default="/share_data/lvjianan/tmp")
    args = parser.parse_args()

    log_file = setup_logging(args.output_dir)
    log_msg(f"Start Processing. Input: {args.root_dir}")

    os.makedirs(args.tmp_dir, exist_ok=True)
    os.environ['TMPDIR'] = args.tmp_dir
    os.environ['TMP'] = args.tmp_dir
    os.environ['TEMP'] = args.tmp_dir

    # 扫描视频
    video_files = natsorted(glob(os.path.join(args.root_dir, "*", "*", "processed", "*_crop*.mp4")))
    total_videos = len(video_files)
    if args.n != -1:
        split_cnt = 55000
        video_files = video_files[(args.n) * split_cnt:(args.n+1) * split_cnt -1] if args.n != 6 else video_files[args.n*split_cnt:total_videos]
    elif args.start_idx > 0 or args.end_idx > 0:
        start_idx = max(0, args.start_idx)
        end_idx = args.end_idx if args.end_idx > 0 else total_videos
        end_idx = min(end_idx, total_videos)
        video_files = video_files[start_idx:end_idx]
    elif args.limit > 0:
        video_files = video_files[:args.limit]

    log_msg(f"Found {len(video_files)} crop videos.")

    if len(video_files) == 0:
        log_msg("No videos to process, exiting.")
        return

    set_start_method('spawn', force=True)
    parent_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', "0,1,2,3,4,5,6,7")
    gpu_list = [x.strip() for x in parent_cuda_visible.split(',')]
    num_gpus = min(args.num_gpus, len(gpu_list))
    total_workers = num_gpus * args.workers_per_gpu
    args_dict = vars(args)

    # 将视频列表均匀分成 total_workers 份
    video_chunks = [[] for _ in range(total_workers)]
    for i, v in enumerate(video_files):
        video_chunks[i % total_workers].append(v)

    log_msg(f"Stage 1: {num_gpus} GPUs x {args.workers_per_gpu} workers/GPU = {total_workers} workers")
    for wi in range(total_workers):
        gpu_idx = wi // args.workers_per_gpu
        log_msg(f"  Worker {wi} -> GPU {gpu_list[gpu_idx]}: {len(video_chunks[wi])} videos")

    import time
    pipeline_start_time = time.time()

    # 启动 worker 进程
    processes = []
    for wi in range(total_workers):
        if len(video_chunks[wi]) == 0:
            continue
        gpu_idx = wi // args.workers_per_gpu
        gpu_id_str = gpu_list[gpu_idx]
        p = Process(
            target=gpu_worker,
            args=(wi, gpu_id_str, video_chunks[wi], args.output_dir, args_dict, log_file)
        )
        processes.append(p)
        p.start()

    # 等待所有 worker 完成
    for p in processes:
        p.join()

    stage1_time = time.time() - pipeline_start_time
    log_msg(f"Stage 1 completed in {stage1_time:.1f}s ({stage1_time/60:.1f} minutes)")
    log_msg("All Stage 1 tasks completed.")

if __name__ == '__main__':
    main()
