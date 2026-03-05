import os
import sys

# ==========================================
# 0. 环境修复
# ==========================================
try:
    conda_prefix = os.environ.get('CONDA_PREFIX')
    if not conda_prefix:
        bin_dir = os.path.dirname(sys.executable)
        if os.path.basename(bin_dir) == 'bin':
            conda_prefix = os.path.dirname(bin_dir)

    if conda_prefix:
        lib_path = os.path.join(conda_prefix, 'lib')
        if os.path.exists(lib_path):
            current_ld = os.environ.get('LD_LIBRARY_PATH', '')
            if lib_path not in current_ld.split(os.pathsep):
                os.environ['LD_LIBRARY_PATH'] = f"{lib_path}{os.pathsep}{current_ld}"
except Exception as e:
    pass

import argparse
import numpy as np
import joblib
from tqdm import tqdm
import traceback
from multiprocessing import Process, set_start_method
import logging
from datetime import datetime
import time
import copy
from glob import glob
from natsort import natsorted
from pathlib import Path
import json

# ==========================================
# 0. 基础配置
# ==========================================
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

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
    log_file = os.path.join(log_dir, f"pipeline_hawor_seq_{timestamp}.log")
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
# 3. 核心处理步骤
# ==========================================

def run_slam_step(video_path, output_root, args_dict):
    """运行 SLAM，返回 (success, start_idx, end_idx)"""
    from scripts.scripts_test_video.hawor_slam import hawor_slam

    args = argparse.Namespace(**copy.deepcopy(args_dict))
    video_id = get_video_id(video_path)
    video_out_dir = os.path.abspath(os.path.join(output_root, video_id))

    track_dirs = glob(os.path.join(video_out_dir, "tracks_*"))
    if not track_dirs:
        log_msg(f"[{video_id}] SKIP SLAM: No tracks found")
        return False, 0, 0

    parts = os.path.basename(track_dirs[0]).split('_')
    start_idx, end_idx = int(parts[1]), int(parts[2])

    slam_path = os.path.join(video_out_dir, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if os.path.exists(slam_path):
        try:
            slam_data = np.load(slam_path)
            if 'tstamp' in slam_data and 'scale' in slam_data:
                return True, start_idx, end_idx
        except:
            pass

    log_msg(f"[{video_id}] Running SLAM...")
    start_time = time.time()

    args.video_path = video_path
    args.seq_folder = video_out_dir

    # 读取内参
    focal = get_focal_length_from_intrinsics(video_path)
    if focal:
        args.img_focal = focal
    else:
        motion_cache = os.path.join(video_out_dir, f"motion_estimation_{start_idx}_{end_idx}.pkl")
        if os.path.exists(motion_cache):
            try:
                m_data = joblib.load(motion_cache)
                args.img_focal = m_data.get('img_focal', None)
            except:
                args.img_focal = None
        else:
            args.img_focal = None

    clear_gpu_memory()

    max_retries = 3
    for retry in range(max_retries):
        try:
            hawor_slam(args, start_idx, end_idx, seq_folder=video_out_dir)
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and retry < max_retries - 1:
                clear_gpu_memory()
                time.sleep(5 * (retry + 1))
            else:
                raise

    clear_gpu_memory()
    log_msg(f"[{video_id}] SLAM finished in {time.time() - start_time:.1f}s")
    return True, start_idx, end_idx


def run_infiller_step(video_path, output_root, args_dict, start_idx, end_idx):
    """运行 Infiller + 保存最终结果"""
    from scripts.scripts_test_video.hawor_video import hawor_infiller
    from lib.eval_utils.custom_utils import load_slam_cam

    args = argparse.Namespace(**copy.deepcopy(args_dict))
    video_id = get_video_id(video_path)
    video_out_dir = os.path.abspath(os.path.join(output_root, video_id))

    result_file = os.path.join(video_out_dir, 'hawor_results.pkl')
    if os.path.exists(result_file):
        return True

    log_msg(f"[{video_id}] Running Infiller...")
    start_time = time.time()

    motion_cache = os.path.join(video_out_dir, f"motion_estimation_{start_idx}_{end_idx}.pkl")
    if not os.path.exists(motion_cache):
        log_msg(f"[{video_id}] SKIP Infiller: Motion cache missing")
        return False

    m_data = joblib.load(motion_cache)
    frame_chunks_all = m_data['frame_chunks_all']
    focal = get_focal_length_from_intrinsics(video_path)
    img_focal = focal if focal else m_data['img_focal']

    slam_path = os.path.join(video_out_dir, f"SLAM/hawor_slam_w_scale_{start_idx}_{end_idx}.npz")
    if not os.path.exists(slam_path):
        log_msg(f"[{video_id}] SKIP Infiller: SLAM result missing")
        return False

    args.video_path = video_path
    args.seq_folder = video_out_dir

    clear_gpu_memory()

    max_retries = 2
    res = None
    for retry in range(max_retries):
        try:
            res = hawor_infiller(args, start_idx, end_idx, frame_chunks_all)
            break
        except RuntimeError as e:
            if "out of memory" in str(e).lower() and retry < max_retries - 1:
                log_msg(f"[{video_id}] Infiller OOM, retrying...")
                clear_gpu_memory()
                time.sleep(2)
            else:
                raise

    if res is None:
        return False

    clear_gpu_memory()

    # 保存结果
    R_w2c, t_w2c, R_c2w, t_c2w = load_slam_cam(slam_path)

    def to_numpy(x):
        import torch
        if isinstance(x, torch.Tensor): return x.cpu().numpy()
        return np.array(x)

    final_res = {
        'video_id': video_id,
        'pred_trans': to_numpy(res[0]),
        'pred_rot': to_numpy(res[1]),
        'pred_hand_pose': to_numpy(res[2]),
        'pred_betas': to_numpy(res[3]),
        'pred_valid': to_numpy(res[4]),
        'slam_R_w2c': to_numpy(R_w2c),
        'slam_t_w2c': to_numpy(t_w2c),
        'slam_R_c2w': to_numpy(R_c2w),
        'slam_t_c2w': to_numpy(t_c2w),
        'img_focal': img_focal
    }
    joblib.dump(final_res, result_file)

    log_msg(f"[{video_id}] Infiller finished in {time.time() - start_time:.1f}s")
    return True


# ==========================================
# 4. GPU Worker 架构
# ==========================================

def gpu_worker(worker_id, gpu_id_str, video_list, output_root, args_dict, log_file_path):
    """
    GPU Worker 进程：
    1. 绑定到指定 GPU
    2. 一次性加载 Metric3D + Infiller 模型
    3. 循环处理分配的所有视频（SLAM → Infiller），复用模型
    """
    setup_subprocess_logging(log_file_path)

    # 绑定 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_id_str

    import torch
    import random

    time.sleep(random.uniform(0.1, 1.0))
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

    # ---- 一次性加载模型 ----
    log_msg(f"[W{worker_id}] GPU {gpu_id_str}: Loading models...")
    load_start = time.time()

    # 1) Metric3D 模型 → 设置到 hawor_slam 模块的全局缓存
    import scripts.scripts_test_video.hawor_slam as hawor_slam_module
    from scripts.scripts_test_video.hawor_slam import get_metric_model
    get_metric_model('thirdparty/Metric3D/weights/metric_depth_vit_large_800k.pth')

    # 2) Infiller 模型 → 设置到 hawor_video 模块的全局缓存
    import scripts.scripts_test_video.hawor_video as hawor_video_module
    from infiller.lib.model.network import TransformerModel
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    weight_path = args_dict['infiller_weight']
    ckpt = torch.load(weight_path, map_location=device)
    pos_dim = 3
    shape_dim = 10
    num_joints = 15
    rot_dim = (num_joints + 1) * 6
    repr_dim = 2 * (pos_dim + shape_dim + rot_dim)
    nhead = 8
    horizon = 120
    filling_model = TransformerModel(
        seq_len=horizon, input_dim=repr_dim, d_model=384, nhead=nhead,
        d_hid=2048, nlayers=8, dropout=0.05, out_dim=repr_dim, masked_attention_stage=True
    )
    filling_model.to(device)
    filling_model.load_state_dict(ckpt['transformer_encoder_state_dict'])
    filling_model.eval()
    hawor_video_module._infiller_model_cache = filling_model

    log_msg(f"[W{worker_id}] GPU {gpu_id_str}: Models loaded in {time.time()-load_start:.1f}s, processing {len(video_list)} videos")

    # ---- 循环处理视频 ----
    for vi, video_path in enumerate(video_list):
        video_id = get_video_id(video_path)
        start_t = time.time()
        try:
            log_msg(f"[W{worker_id}][{vi+1}/{len(video_list)}][{video_id}] Pipeline Started")

            # 1. SLAM
            success, start_idx, end_idx = run_slam_step(video_path, output_root, args_dict)
            if not success:
                continue

            # 2. Infiller
            run_infiller_step(video_path, output_root, args_dict, start_idx, end_idx)

            total_time = time.time() - start_t
            log_msg(f"[W{worker_id}][{vi+1}/{len(video_list)}][{video_id}] Pipeline Completed in {total_time:.1f}s")

        except Exception as e:
            log_msg(f"[{video_id}] ERROR: {e}", level="error")
            log_msg(traceback.format_exc(), level="error")


# ==========================================
# 5. 主程序
# ==========================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root_dir", type=str, default='/share_data/lvjianan/datasets/BuildAI-processed')
    parser.add_argument("--output_dir", type=str, default='/share_data/lvjianan/HaWoR/output/buildai_results')
    parser.add_argument("--checkpoint", type=str, default='/share_data/lvjianan/HaWoR/weights/hawor/checkpoints/hawor.ckpt')
    parser.add_argument("--infiller_weight", type=str, default='/share_data/lvjianan/HaWoR/weights/hawor/checkpoints/infiller.pt')
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--workers_per_gpu", type=int, default=1, help="Number of worker processes per GPU")
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=-1)
    parser.add_argument("--tmp_dir", type=str, default="/share_data/lvjianan/tmp")
    args = parser.parse_args()

    log_file = setup_logging(args.output_dir)
    log_msg("=" * 80)
    log_msg("HaWoR Pipeline: SLAM + Infiller (GPU Worker Architecture)")
    log_msg("=" * 80)

    tmp_dir = args.tmp_dir
    os.makedirs(tmp_dir, exist_ok=True)
    os.environ['TMPDIR'] = tmp_dir
    os.environ['TMP'] = tmp_dir
    os.environ['TEMP'] = tmp_dir

    # 扫描视频
    video_files = natsorted(glob(os.path.join(args.root_dir, "*", "*", "processed", "*_crop*.mp4")))
    total_videos = len(video_files)

    if args.start_idx > 0 or args.end_idx > 0:
        start_idx = max(0, args.start_idx)
        end_idx = min(args.end_idx if args.end_idx > 0 else total_videos, total_videos)
        video_files = video_files[start_idx:end_idx]
        log_msg(f"Video range: [{start_idx}:{end_idx}]")
    elif args.limit > 0:
        video_files = video_files[:args.limit]

    log_msg(f"Total videos to process: {len(video_files)}")

    if len(video_files) == 0:
        log_msg("No videos to process, exiting.")
        return

    # GPU 资源分配
    parent_cuda_visible = os.environ.get('CUDA_VISIBLE_DEVICES', ','.join(str(i) for i in range(args.num_gpus)))
    gpu_list = [x.strip() for x in parent_cuda_visible.split(',')]
    num_gpus = min(args.num_gpus, len(gpu_list))
    total_workers = num_gpus * args.workers_per_gpu

    # 将视频列表均匀分成 total_workers 份
    video_chunks = [[] for _ in range(total_workers)]
    for i, v in enumerate(video_files):
        video_chunks[i % total_workers].append(v)

    log_msg(f"Resources: {num_gpus} GPUs x {args.workers_per_gpu} workers/GPU = {total_workers} workers")
    for wi in range(total_workers):
        gpu_idx = wi // args.workers_per_gpu
        log_msg(f"  Worker {wi} -> GPU {gpu_list[gpu_idx]}: {len(video_chunks[wi])} videos")

    set_start_method('spawn', force=True)
    start_time = time.time()

    # 启动 worker 进程
    processes = []
    for wi in range(total_workers):
        if len(video_chunks[wi]) == 0:
            continue
        gpu_idx = wi // args.workers_per_gpu
        gpu_id_str = gpu_list[gpu_idx]
        p = Process(
            target=gpu_worker,
            args=(wi, gpu_id_str, video_chunks[wi], args.output_dir, vars(args), log_file)
        )
        processes.append(p)
        p.start()

    # 等待所有 worker 完成
    for p in processes:
        p.join()

    total_time = time.time() - start_time
    log_msg(f"\nAll Completed in {total_time/60:.1f} minutes")

if __name__ == '__main__':
    main()
