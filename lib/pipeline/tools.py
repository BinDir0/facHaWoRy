import cv2
from tqdm import tqdm
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor

from ultralytics import YOLO


if torch.cuda.is_available():
    autocast = torch.cuda.amp.autocast
else:
    class autocast:
        def __init__(self, enabled=True):
            pass
        def __enter__(self):
            pass
        def __exit__(self, *args):
            pass


# Global model cache to avoid reloading
_hand_det_model_cache = None

def detect_track(imgfiles, thresh=0.4, batch_size=16, model=None, chunk_size=500):
    """
    内存优化版 detect_track：
    1. 分批处理图像列表，避免一次性处理太多导致 OOM
    2. 每批之间清理 GPU 内存
    3. 使用 stream=True 节省内存
    """
    global _hand_det_model_cache
    if model is not None:
        hand_det_model = model
    elif _hand_det_model_cache is not None:
        hand_det_model = _hand_det_model_cache
    else:
        from ultralytics import YOLO
        import os
        # Try to find the weights file relative to this file
        current_dir = os.path.dirname(os.path.abspath(__file__))
        # HaWoR/lib/pipeline/../../weights/external/detector.pt -> HaWoR/weights/external/detector.pt
        weights_path = os.path.join(current_dir, '..', '..', 'weights', 'external', 'detector.pt')
        weights_path = os.path.abspath(weights_path)
        
        if not os.path.exists(weights_path):
             # Fallback to relative path if not found (e.g. if running from HaWoR root)
             weights_path = './weights/external/detector.pt'
             
        hand_det_model = YOLO(weights_path)
        if torch.cuda.is_available():
            hand_det_model.to('cuda')
        _hand_det_model_cache = hand_det_model

    tracks = {}
    total_frames = len(imgfiles)
    
    # 分批处理，避免内存溢出
    num_chunks = (total_frames + chunk_size - 1) // chunk_size
    
    for chunk_idx in tqdm(range(num_chunks), desc="Detecting and tracking"):
        start_idx = chunk_idx * chunk_size
        end_idx = min(start_idx + chunk_size, total_frames)
        chunk_files = imgfiles[start_idx:end_idx]
        
        # 清理 GPU 缓存（在每批开始前）
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # 使用 stream=True 处理这一批图像
        results_generator = hand_det_model.track(
            source=chunk_files,
            conf=thresh,
            persist=True,
            stream=True,     # 生成器模式，节省内存
            verbose=False,
            imgsz=512,       # 降低分辨率以节省内存
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )

        for local_t, r in enumerate(results_generator):
            t = start_idx + local_t  # 全局帧索引
            if r.boxes is None or r.boxes.id is None:
                continue
            
            # 批量获取数据，减少转化开销
            ids = r.boxes.id.int().cpu().tolist()
            boxes = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            clss = r.boxes.cls.int().cpu().tolist()

            find_right = False
            find_left = False

            for i in range(len(ids)):
                track_id = ids[i]
                handedness = clss[i]
                
                # 过滤逻辑 (保持你原有的逻辑)
                if (not find_right and handedness > 0) or (not find_left and handedness == 0):
                    subj = {
                        'frame': t,
                        'det': True,
                        'det_box': np.append(boxes[i], confs[i])[None, :],
                        'det_handedness': np.array([handedness])
                    }
                    
                    if track_id not in tracks:
                        tracks[track_id] = []
                    tracks[track_id].append(subj)

                    if handedness > 0: find_right = True
                    else: find_left = True
        
        # 每批处理完后清理内存
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # 最终清理
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return None, tracks # boxes_ 在你后面的代码里好像没用到


def parse_chunks(frame, boxes, min_len=16):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    boxes_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        b_chunk = boxes[start:bk]
        start = bk
        if len(f_chunk)>=min_len:
            frame_chunks.append(f_chunk)
            boxes_chunks.append(b_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            b_chunk = boxes[bk:]
            if len(f_chunk)>=min_len:
                frame_chunks.append(f_chunk)
                boxes_chunks.append(b_chunk)

    return frame_chunks, boxes_chunks

def parse_chunks_hand_frame(frame):
    """ If a track disappear in the middle, 
     we separate it to different segments to estimate the HPS independently. 
     If a segment is less than 16 frames, we get rid of it for now. 
     """
    frame_chunks = []
    step = frame[1:] - frame[:-1]
    step = np.concatenate([[0], step])
    breaks = np.where(step != 1)[0]

    start = 0
    for bk in breaks:
        f_chunk = frame[start:bk]
        start = bk
        if len(f_chunk) > 0:
            frame_chunks.append(f_chunk)

        if bk==breaks[-1]:  # last chunk
            f_chunk = frame[bk:]
            if len(f_chunk) > 0:
                frame_chunks.append(f_chunk)

    return frame_chunks
