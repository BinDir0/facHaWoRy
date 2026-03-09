import argparse
import numpy as np
import os
import sys

sys.path.insert(0, os.path.dirname(__file__) + '/../..')

from lib.pipeline.frame_source import build_frame_source_auto
from lib.pipeline.tools import detect_track

# Check if we should suppress verbose output
QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"

def vprint(*args, **kwargs):
    """Print only if not in quiet mode."""
    if not QUIET_MODE:
        print(*args, **kwargs)


def detect_track_video(args, detector_runner=None, force=False, detect_batch_size=1, prefetch_frames=16, device='cuda:0', half_precision=True):
    file = args.video_path
    root = os.path.dirname(file)
    seq = os.path.basename(file).split('.')[0]

    seq_folder = f'{root}/{seq}'
    os.makedirs(seq_folder, exist_ok=True)
    vprint(f'Running detect_track on {file} ...')

    backend = getattr(args, 'frame_backend', 'decord')
    frame_source, backend_used = build_frame_source_auto(file, backend=backend)
    vprint(f'Frame backend: {backend_used}')

    ##### Detection + Track #####
    vprint('Detect and Track ...')

    start_idx = 0
    end_idx = len(frame_source)

    if (not force) and os.path.exists(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy'):
        vprint(f"skip track for {start_idx}_{end_idx}")
        return start_idx, end_idx, seq_folder, frame_source

    # Invalidate track range cache since we're (re)running detect_track
    cache_file = f'{seq_folder}/.track_range'
    if os.path.exists(cache_file):
        os.remove(cache_file)

    os.makedirs(f"{seq_folder}/tracks_{start_idx}_{end_idx}", exist_ok=True)
    # Increased threshold from 0.2 to 0.35 to reduce false positives
    # Especially important when hands leave frame or camera moves rapidly
    # Edge detections require even higher confidence (0.4) to avoid background objects
    boxes_, tracks_ = detect_track(
        frame_source,
        thresh=0.35,
        edge_margin_ratio=0.1,
        min_edge_conf=0.4,
        hand_det_model=detector_runner,
        reset_tracker=True,
        detect_batch_size=detect_batch_size,
        prefetch_frames=prefetch_frames,
        device=device,
        half_precision=half_precision,
    )
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_boxes.npy', boxes_)
    np.save(f'{seq_folder}/tracks_{start_idx}_{end_idx}/model_tracks.npy', tracks_)

    return start_idx, end_idx, seq_folder, frame_source


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--video_path", type=str, default='')
    parser.add_argument("--input_type", type=str, default='file')
    parser.add_argument("--frame_backend", type=str, default='decord', choices=['decord', 'opencv'])
    args = parser.parse_args()

    detect_track_video(args)
