import argparse
import json
import os
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


STAGES = ["detect_track", "motion", "slam", "infiller"]


def set_determinism(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_seq_folder(video_path: str) -> Path:
    video_path = Path(video_path)
    return video_path.parent / video_path.stem


def get_track_range(seq_folder: Path, fast=False):
    """
    Get track range from seq_folder.

    Args:
        seq_folder: Sequence folder path
        fast: If True, assume tracks_0_N format (faster, no glob)
    """
    if fast:
        # Fast mode: assume standard naming tracks_0_N
        # Try to find it without globbing
        for p in seq_folder.iterdir():
            if p.is_dir() and p.name.startswith("tracks_0_"):
                parts = p.name.split("_")
                if len(parts) == 3:
                    try:
                        start_idx = int(parts[1])
                        end_idx = int(parts[2])
                        return start_idx, end_idx
                    except ValueError:
                        pass
        # Fallback to slow method if fast fails

    # Slow method: glob all tracks_*_* directories
    track_dirs = []
    for p in seq_folder.glob("tracks_*_*"):
        parts = p.name.split("_")
        if len(parts) != 3:
            continue
        try:
            start_idx = int(parts[1])
            end_idx = int(parts[2])
        except ValueError:
            continue
        track_dirs.append((start_idx, end_idx, p))

    if not track_dirs:
        raise FileNotFoundError(f"No tracks_*_* folder found under {seq_folder}")

    track_dirs.sort(key=lambda x: (x[1], x[0]))
    start_idx, end_idx, _ = track_dirs[-1]
    return start_idx, end_idx


def validate_stage_output(stage: str, seq_folder: Path, start_idx: int, end_idx: int):
    tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"

    if stage == "detect_track":
        assert (tracks_dir / "model_boxes.npy").exists(), "model_boxes.npy missing"
        assert (tracks_dir / "model_tracks.npy").exists(), "model_tracks.npy missing"
        img_dir = seq_folder / "extracted_images"
        assert img_dir.exists(), "extracted_images missing"
        assert len(list(img_dir.glob("*.jpg"))) > 0, "no extracted frames"
        return

    if stage == "motion":
        assert (tracks_dir / "frame_chunks_all.npy").exists(), "frame_chunks_all.npy missing"
        assert (tracks_dir / "model_masks.npy").exists(), "model_masks.npy missing"
        model_masks = np.load(tracks_dir / "model_masks.npy", allow_pickle=True)
        assert model_masks.ndim == 3, "model_masks should be (T,H,W)"
        return

    if stage == "slam":
        slam_file = seq_folder / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
        assert slam_file.exists(), "SLAM npz missing"
        data = np.load(slam_file, allow_pickle=True)
        assert "traj" in data and "scale" in data, "invalid SLAM npz keys"
        return

    if stage == "infiller":
        world_file = seq_folder / "world_space_res.pth"
        assert world_file.exists(), "world_space_res.pth missing"
        pred_trans, pred_rot, pred_hand_pose, pred_betas, pred_valid = joblib.load(world_file)
        assert pred_trans.shape[0] == 2 and pred_trans.shape[-1] == 3, "pred_trans shape invalid"
        assert pred_rot.shape[0] == 2 and pred_rot.shape[-1] == 3, "pred_rot shape invalid"
        assert pred_hand_pose.shape[0] == 2 and pred_hand_pose.shape[-1] == 45, "pred_hand_pose shape invalid"
        assert pred_betas.shape[0] == 2 and pred_betas.shape[-1] == 10, "pred_betas shape invalid"
        assert pred_valid.shape[0] == 2, "pred_valid shape invalid"
        return

    raise ValueError(f"Unknown stage: {stage}")


def is_stage_complete(stage: str, seq_folder: Path, fast_check=False):
    """
    Check if a stage is complete.

    Args:
        stage: Stage name
        seq_folder: Sequence folder path
        fast_check: If True, only check file existence without loading/validating content
    """
    try:
        start_idx, end_idx = get_track_range(seq_folder, fast=fast_check)

        if fast_check:
            # Fast check: only verify files exist, don't load them
            return validate_stage_output_fast(stage, seq_folder, start_idx, end_idx)
        else:
            # Full validation: load and check content
            validate_stage_output(stage, seq_folder, start_idx, end_idx)
            return True
    except Exception:
        return False


def validate_stage_output_fast(stage: str, seq_folder: Path, start_idx: int, end_idx: int):
    """Fast validation: only check if required files exist."""
    tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"

    if stage == "detect_track":
        return (
            (tracks_dir / "model_boxes.npy").exists() and
            (tracks_dir / "model_tracks.npy").exists() and
            (seq_folder / "extracted_images").exists()
        )

    if stage == "motion":
        return (
            (tracks_dir / "frame_chunks_all.npy").exists() and
            (tracks_dir / "model_masks.npy").exists()
        )

    if stage == "slam":
        slam_file = seq_folder / "SLAM" / f"hawor_slam_w_scale_{start_idx}_{end_idx}.npz"
        return slam_file.exists()

    if stage == "infiller":
        world_res = seq_folder / "world_space_res.pth"
        return world_res.exists()

    return False


def build_stage_args(ns):
    class StageArgs:
        pass

    args = StageArgs()
    args.img_focal = ns.img_focal
    args.video_path = ns.video_path
    args.input_type = ns.input_type
    args.checkpoint = ns.checkpoint
    args.infiller_weight = ns.infiller_weight
    args.vis_mode = "world"
    args.skip_vis = True
    return args


def emit_event(event: str, **kwargs):
    payload = {
        "time": datetime.now(timezone.utc).isoformat(),
        "event": event,
        **kwargs,
    }
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run_stage(ns):
    if ns.gpu is not None and ns.gpu != "":
        os.environ["CUDA_VISIBLE_DEVICES"] = str(ns.gpu)

    set_determinism(ns.seed)

    stage_args = build_stage_args(ns)
    seq_folder = get_seq_folder(ns.video_path)

    if ns.resume and not ns.force and is_stage_complete(ns.stage, seq_folder):
        return {
            "status": "skipped",
            "reason": "existing_valid_output",
        }

    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from scripts.scripts_test_video.hawor_video import hawor_infiller, hawor_motion_estimation

    if ns.stage == "detect_track":
        start_idx, end_idx, _, _ = detect_track_video(stage_args)
    else:
        start_idx, end_idx = get_track_range(seq_folder)

    if ns.stage == "motion":
        hawor_motion_estimation(stage_args, start_idx, end_idx, str(seq_folder))
    elif ns.stage == "slam":
        hawor_slam(stage_args, start_idx, end_idx)
    elif ns.stage == "infiller":
        tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
        frame_chunks_all = joblib.load(tracks_dir / "frame_chunks_all.npy")
        hawor_infiller(stage_args, start_idx, end_idx, frame_chunks_all)
    elif ns.stage == "detect_track":
        pass
    else:
        raise ValueError(f"Unknown stage: {ns.stage}")

    validate_stage_output(ns.stage, seq_folder, start_idx, end_idx)
    return {
        "status": "success",
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--video_path", required=True, type=str)
    parser.add_argument("--gpu", default="", type=str)
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true", help="Ignore existing outputs and rerun this stage")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()
    started_at = time.time()
    common_fields = {
        "video": args.video_path,
        "stage": args.stage,
        "gpu": args.gpu,
    }

    emit_event("stage_start", **common_fields)
    try:
        result = run_stage(args)
        emit_event(
            "stage_end",
            **common_fields,
            status=result.get("status", "success"),
            elapsed_sec=round(time.time() - started_at, 3),
            reason=result.get("reason"),
            start_idx=result.get("start_idx"),
            end_idx=result.get("end_idx"),
        )
        sys.exit(0)
    except Exception as err:
        emit_event(
            "stage_end",
            **common_fields,
            status="failed",
            elapsed_sec=round(time.time() - started_at, 3),
            error=str(err),
        )
        traceback.print_exc()
        sys.exit(1)
