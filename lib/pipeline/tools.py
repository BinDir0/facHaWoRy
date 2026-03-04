import cv2
from tqdm import tqdm
import numpy as np
import torch

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


def detect_track(frame_source, thresh=0.5, edge_margin_ratio=0.1, min_edge_conf=0.4):
    """
    Detect and track hands in video frames.

    Args:
        frame_source: Frame source object that yields BGR frames
        thresh: Base confidence threshold for detection
        edge_margin_ratio: Ratio of image size to define edge region (default 0.1 = 10%)
        min_edge_conf: Minimum confidence required for detections near edges
    """
    hand_det_model = YOLO('./weights/external/detector.pt')

    boxes_ = []
    tracks = {}
    fallback_counter = 0  # Global counter for unique fallback IDs
    track_last_seen = {}  # Track when each track was last seen with good quality

    for t, img_cv2 in tqdm(frame_source.iter_frames(rgb=False), total=len(frame_source)):
        img_h, img_w = img_cv2.shape[:2]

        # Define edge regions
        edge_left = img_w * edge_margin_ratio
        edge_right = img_w * (1 - edge_margin_ratio)
        edge_top = img_h * edge_margin_ratio
        edge_bottom = img_h * (1 - edge_margin_ratio)

        ### --- Detection ---
        with torch.no_grad():
            with autocast():
                results = hand_det_model.track(img_cv2, conf=thresh, persist=True, verbose=False)

                boxes = results[0].boxes.xyxy.cpu().numpy()
                confs = results[0].boxes.conf.cpu().numpy()
                handedness = results[0].boxes.cls.cpu().numpy()
                if not results[0].boxes.id is None:
                    track_id = results[0].boxes.id.cpu().numpy()
                else:
                    track_id = [-1] * len(boxes)

                boxes = np.hstack([boxes, confs[:, None]])
                boxes_.append(boxes)
                find_right = False
                find_left = False
                for idx, box in enumerate(boxes):
                    # Check if detection is near edge
                    x1, y1, x2, y2, conf = boxes[idx]
                    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                    is_near_edge = (cx < edge_left or cx > edge_right or
                                   cy < edge_top or cy > edge_bottom)

                    # Apply stricter confidence threshold for edge detections
                    if is_near_edge and conf < min_edge_conf:
                        continue  # Skip low-confidence edge detections

                    if track_id[idx] == -1:
                        # Generate unique fallback ID using global counter
                        # This prevents multiple untracked detections from getting the same ID
                        if handedness[[idx]] > 0:
                            id = int(10000 + fallback_counter)  # Right hand fallbacks: 10000+
                        else:
                            id = int(5000 + fallback_counter)   # Left hand fallbacks: 5000+
                        fallback_counter += 1
                    else:
                        id = track_id[idx]

                    # Check track quality: if track hasn't been seen for a while, be more strict
                    if id in track_last_seen:
                        frames_since_last = t - track_last_seen[id]
                        if frames_since_last > 10:  # Track was lost for >10 frames
                            # Require higher confidence to resume track
                            if conf < min_edge_conf:
                                continue

                    subj = dict()
                    subj['frame'] = t
                    subj['det'] = True
                    subj['det_box'] = boxes[[idx]]
                    subj['det_handedness'] = handedness[[idx]]
                    subj['is_near_edge'] = is_near_edge


                    if (not find_right and handedness[[idx]] > 0) or (not find_left and handedness[[idx]]==0):
                        if id in tracks:
                            tracks[id].append(subj)
                        else:
                            tracks[id] = [subj]

                        track_last_seen[id] = t  # Update last seen time

                        if handedness[[idx]] > 0:
                            find_right = True
                        elif handedness[[idx]] == 0:
                            find_left = True
    tracks = np.array(tracks, dtype=object)
    boxes_ = np.array(boxes_, dtype=object)

    return boxes_, tracks


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
