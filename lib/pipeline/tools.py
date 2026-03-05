import cv2
from tqdm import tqdm
import numpy as np
import torch
import os

from ultralytics import YOLO

# Check if we should suppress verbose output
QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"


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


def detect_track(frame_source, thresh=0.5, edge_margin_ratio=0.1, min_edge_conf=0.4, hand_det_model=None, reset_tracker=True, detect_batch_size=1):
    """
    Detect and track hands in video frames.

    Args:
        frame_source: Frame source object that yields BGR frames
        thresh: Base confidence threshold for detection
        edge_margin_ratio: Ratio of image size to define edge region (default 0.1 = 10%)
        min_edge_conf: Minimum confidence required for detections near edges
        hand_det_model: Optional preloaded YOLO detector for reuse across videos
        reset_tracker: Reset tracker state before running current video
        detect_batch_size: MUST be 1 - kept for API compatibility but frame-level batching is not supported

    CRITICAL NOTE:
        Tracking is inherently sequential and stateful. Each frame's tracking depends on
        the previous frame's Kalman filter state. Batching multiple frames from the same
        video breaks this state dependency and causes "LinAlgError: leading minor not
        positive definite" errors in the tracker.

        For performance optimization, use cross-video batching at the scheduler level
        (multiple GPUs processing different videos in parallel) instead of frame-level batching.
    """
    # Validate detect_batch_size
    if detect_batch_size != 1:
        raise ValueError(
            f"detect_batch_size must be 1 (got {detect_batch_size}). "
            "Frame-level batching breaks YOLO tracker state."
        )

    hand_det_model = hand_det_model or YOLO('./weights/external/detector.pt')

    if reset_tracker and hasattr(hand_det_model, 'predictor') and hand_det_model.predictor is not None:
        hand_det_model.predictor = None

    boxes_ = []
    tracks = {}
    fallback_counter = 0  # Global counter for unique fallback IDs
    track_last_seen = {}  # Track when each track was last seen with good quality

    # Process frames sequentially to maintain tracker state
    for t, img_cv2 in tqdm(frame_source.iter_frames(rgb=False), total=len(frame_source), disable=QUIET_MODE, desc="Detect & Track"):
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


def detect_track_multivideo(video_sources, thresh=0.5, edge_margin_ratio=0.1, min_edge_conf=0.4, hand_det_model=None):
    """
    Process multiple videos with batch inference and independent tracking using Supervision library.

    This function enables cross-video batching: frames from different videos are batched together
    for GPU inference, while each video maintains its own independent tracker state.

    Args:
        video_sources: List of (video_idx, frame_source) tuples
        thresh: Base confidence threshold for detection
        edge_margin_ratio: Ratio of image size to define edge region
        min_edge_conf: Minimum confidence required for detections near edges
        hand_det_model: Shared YOLO model (used for detection only)

    Returns:
        Dict[video_idx, (boxes, tracks)] - Results for each video
    """
    try:
        import supervision as sv
    except ImportError:
        raise ImportError(
            "Supervision library is required for multi-video batch processing. "
            "Install it with: pip install supervision"
        )

    hand_det_model = hand_det_model or YOLO('./weights/external/detector.pt')

    # Create independent ByteTrack tracker for each video
    trackers = {idx: sv.ByteTrack() for idx, _ in video_sources}

    # Initialize state for each video
    video_states = {}
    for idx, fs in video_sources:
        video_states[idx] = {
            'boxes': [],
            'tracks': {},
            'frame_iter': iter(fs.iter_frames(rgb=False)),
            'active': True,
            'fallback_counter': 0,
            'track_last_seen': {},
            'frame_count': 0
        }

    active_videos = [idx for idx, _ in video_sources]

    # Progress bar for overall progress
    total_frames = sum(len(fs) for _, fs in video_sources)
    pbar = tqdm(total=total_frames, disable=QUIET_MODE, desc="Detect & Track (Multi-Video)")

    # Performance timing
    import time
    from concurrent.futures import ThreadPoolExecutor
    io_time = 0.0
    gpu_time = 0.0
    tracker_time = 0.0
    batch_count = 0

    while active_videos:
        # Collect current frame from each active video
        batch_frames = []
        batch_meta = []  # (video_idx, frame_t, img_shape)

        # TIME: I/O - Frame reading (parallel)
        t0 = time.time()

        # Parallel frame reading using ThreadPoolExecutor
        def read_frame(vid_idx):
            try:
                t, img_cv2 = next(video_states[vid_idx]['frame_iter'])
                return (vid_idx, t, img_cv2, img_cv2.shape[:2], None)
            except StopIteration:
                return (vid_idx, None, None, None, 'stop')

        with ThreadPoolExecutor(max_workers=len(active_videos)) as executor:
            futures = [executor.submit(read_frame, vid_idx) for vid_idx in active_videos[:]]
            results = [f.result() for f in futures]

        # Process results
        for vid_idx, t, img_cv2, img_shape, error in results:
            if error == 'stop':
                video_states[vid_idx]['active'] = False
                active_videos.remove(vid_idx)
            else:
                batch_frames.append(img_cv2)
                batch_meta.append((vid_idx, t, img_shape))

        io_time += time.time() - t0

        if not batch_frames:
            break

        # Batch inference (detection only, no tracking)
        # TIME: GPU inference
        t0 = time.time()
        with torch.no_grad():
            with autocast():
                results_batch = hand_det_model.predict(
                    batch_frames,
                    conf=thresh,
                    verbose=False
                )
        gpu_time += time.time() - t0

        # Process results for each video independently
        # TIME: Tracker update
        t0 = time.time()
        for (vid_idx, t, img_shape), results in zip(batch_meta, results_batch):
            img_h, img_w = img_shape
            state = video_states[vid_idx]

            # Define edge regions
            edge_left = img_w * edge_margin_ratio
            edge_right = img_w * (1 - edge_margin_ratio)
            edge_top = img_h * edge_margin_ratio
            edge_bottom = img_h * (1 - edge_margin_ratio)

            # Convert to supervision format
            detections = sv.Detections.from_ultralytics(results)

            # Update tracker for this video
            tracked_detections = trackers[vid_idx].update_with_detections(detections)

            # Extract results
            boxes = tracked_detections.xyxy
            confs = tracked_detections.confidence
            track_ids = tracked_detections.tracker_id if tracked_detections.tracker_id is not None else np.array([-1] * len(boxes))
            class_ids = tracked_detections.class_id if tracked_detections.class_id is not None else np.array([0] * len(boxes))

            # Store boxes with confidence
            boxes_with_conf = np.hstack([boxes, confs[:, None]]) if len(boxes) > 0 else np.array([])
            state['boxes'].append(boxes_with_conf)

            # Process tracks (same logic as original detect_track)
            find_right = False
            find_left = False

            for idx in range(len(boxes)):
                x1, y1, x2, y2 = boxes[idx]
                conf = confs[idx]
                track_id = track_ids[idx]
                handedness = class_ids[idx]

                # Check if detection is near edge
                cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
                is_near_edge = (cx < edge_left or cx > edge_right or
                               cy < edge_top or cy > edge_bottom)

                # Apply stricter confidence threshold for edge detections
                if is_near_edge and conf < min_edge_conf:
                    continue

                # Handle untracked detections
                if track_id == -1:
                    if handedness > 0:
                        id = int(10000 + state['fallback_counter'])
                    else:
                        id = int(5000 + state['fallback_counter'])
                    state['fallback_counter'] += 1
                else:
                    id = int(track_id)

                # Check track quality
                if id in state['track_last_seen']:
                    frames_since_last = t - state['track_last_seen'][id]
                    if frames_since_last > 10:
                        if conf < min_edge_conf:
                            continue

                subj = {
                    'frame': t,
                    'det': True,
                    'det_box': np.array([[x1, y1, x2, y2, conf]]),
                    'det_handedness': np.array([handedness]),
                    'is_near_edge': is_near_edge
                }

                # Only keep one detection per hand type
                if (not find_right and handedness > 0) or (not find_left and handedness == 0):
                    if id in state['tracks']:
                        state['tracks'][id].append(subj)
                    else:
                        state['tracks'][id] = [subj]

                    state['track_last_seen'][id] = t

                    if handedness > 0:
                        find_right = True
                    elif handedness == 0:
                        find_left = True

            state['frame_count'] += 1
            pbar.update(1)

        tracker_time += time.time() - t0
        batch_count += 1

    pbar.close()

    # Print performance statistics
    total_time = io_time + gpu_time + tracker_time
    print("\n" + "="*60)
    print("Performance Breakdown (detect_track_multivideo):")
    print("="*60)
    print(f"Total batches processed: {batch_count}")
    print(f"Videos per batch: {len(video_sources)}")
    print(f"Total frames: {total_frames}")
    print(f"\nTime breakdown:")
    print(f"  I/O (frame reading):  {io_time:7.2f}s ({io_time/total_time*100:5.1f}%)")
    print(f"  GPU (inference):      {gpu_time:7.2f}s ({gpu_time/total_time*100:5.1f}%)")
    print(f"  Tracker (update):     {tracker_time:7.2f}s ({tracker_time/total_time*100:5.1f}%)")
    print(f"  Total:                {total_time:7.2f}s")
    print(f"\nPer-batch averages:")
    print(f"  I/O per batch:        {io_time/batch_count*1000:6.1f}ms")
    print(f"  GPU per batch:        {gpu_time/batch_count*1000:6.1f}ms")
    print(f"  Tracker per batch:    {tracker_time/batch_count*1000:6.1f}ms")
    print(f"\nGPU utilization: {gpu_time/total_time*100:.1f}%")
    print("="*60 + "\n")

    # Convert results to numpy arrays (same format as original detect_track)
    results = {}
    for vid_idx, state in video_states.items():
        tracks = np.array(state['tracks'], dtype=object)
        boxes = np.array(state['boxes'], dtype=object)
        results[vid_idx] = (boxes, tracks)

    return results


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
