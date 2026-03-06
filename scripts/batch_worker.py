import argparse
import json
import os
import random
import sys
import tempfile
import time
import traceback
import warnings
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import torch

# Suppress common warnings to reduce output noise
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)
warnings.filterwarnings('ignore', message='.*pkg_resources.*')
warnings.filterwarnings('ignore', message='.*timm.models.layers.*')
warnings.filterwarnings('ignore', message='.*torch.cuda.amp.autocast.*')

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Set temporary directory to shared storage instead of local /tmp
# IMPORTANT: Set this AFTER importing torch to avoid library loading issues
SHARED_TMP_DIR = Path("/share_data/guantianrui/tmp")
SHARED_TMP_DIR.mkdir(parents=True, exist_ok=True)
os.environ["TMPDIR"] = str(SHARED_TMP_DIR)
os.environ["TEMP"] = str(SHARED_TMP_DIR)
os.environ["TMP"] = str(SHARED_TMP_DIR)
tempfile.tempdir = str(SHARED_TMP_DIR)

# Suppress verbose output from stage scripts
os.environ["HAWOR_QUIET"] = "1"


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
        fast: If True, use ultra-fast method (read from cache file)
    """
    if fast:
        # Ultra-fast mode: read from .track_range cache file
        cache_file = seq_folder / ".track_range"
        if cache_file.exists():
            try:
                content = cache_file.read_text().strip()
                start_idx, end_idx = map(int, content.split(","))
                return start_idx, end_idx
            except:
                pass  # Fallback to directory scan

        # Fast mode: assume standard naming tracks_0_N
        # Try to find it without full iteration
        for p in seq_folder.iterdir():
            if p.is_dir() and p.name.startswith("tracks_0_"):
                parts = p.name.split("_")
                if len(parts) == 3:
                    try:
                        start_idx = int(parts[1])
                        end_idx = int(parts[2])
                        # Cache for next time
                        cache_file.write_text(f"{start_idx},{end_idx}")
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

    # Cache the result
    cache_file = seq_folder / ".track_range"
    cache_file.write_text(f"{start_idx},{end_idx}")

    return start_idx, end_idx


def validate_stage_output(stage: str, seq_folder: Path, start_idx: int, end_idx: int):
    tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"

    if stage == "detect_track":
        assert (tracks_dir / "model_boxes.npy").exists(), "model_boxes.npy missing"
        assert (tracks_dir / "model_tracks.npy").exists(), "model_tracks.npy missing"
        return

    if stage == "motion":
        # Check for incomplete outputs and auto-fix
        frame_chunks_file = tracks_dir / "frame_chunks_all.npy"
        model_masks_file = tracks_dir / "model_masks.npy"

        if frame_chunks_file.exists() and not model_masks_file.exists():
            # Incomplete output detected - remove to force re-run
            import sys
            print(f"Warning: Incomplete motion output detected for {seq_folder}", file=sys.stderr)
            print(f"  - frame_chunks_all.npy exists but model_masks.npy missing", file=sys.stderr)
            print(f"  - Removing incomplete output to force re-run", file=sys.stderr)
            frame_chunks_file.unlink()
            raise AssertionError("Incomplete motion output - removed and will retry")

        assert frame_chunks_file.exists(), "frame_chunks_all.npy missing"
        assert model_masks_file.exists(), "model_masks.npy missing"
        model_masks = np.load(model_masks_file, allow_pickle=True)
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
        fast_check: If True, use ultra-fast check (only check .done marker file)
    """
    if fast_check:
        # Ultra-fast check: only check .done marker file
        done_marker = seq_folder / f".{stage}.done"
        if done_marker.exists():
            return True
        # If no marker, fall through to file existence check

    try:
        start_idx, end_idx = get_track_range(seq_folder, fast=fast_check)

        if fast_check:
            # Fast check: only verify files exist, don't load them
            result = validate_stage_output_fast(stage, seq_folder, start_idx, end_idx)
            # Create .done marker for next time
            if result:
                done_marker = seq_folder / f".{stage}.done"
                done_marker.touch()
            return result
        else:
            # Full validation: load and check content
            validate_stage_output(stage, seq_folder, start_idx, end_idx)
            # Create .done marker
            done_marker = seq_folder / f".{stage}.done"
            done_marker.touch()
            return True
    except Exception:
        return False


def validate_stage_output_fast(stage: str, seq_folder: Path, start_idx: int, end_idx: int):
    """Fast validation: only check if required files exist."""
    tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"

    if stage == "detect_track":
        return (
            (tracks_dir / "model_boxes.npy").exists() and
            (tracks_dir / "model_tracks.npy").exists()
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


class WorkerRuntime:
    def __init__(
        self,
        gpu: str,
        checkpoint: str,
        infiller_weight: str,
        img_focal: float = None,
        input_type: str = "file",
        chunk_batch_size: int = 4,
        metric3d_batch_size: int = 8,
        detect_batch_size: int = 8,
        frame_backend: str = "decord",
    ):
        self.gpu = gpu
        self.checkpoint = checkpoint
        self.infiller_weight = infiller_weight
        self.img_focal = img_focal
        self.input_type = input_type
        self.chunk_batch_size = chunk_batch_size
        self.metric3d_batch_size = metric3d_batch_size
        self.detect_batch_size = detect_batch_size
        self.frame_backend = frame_backend

        self.detector_runner = None
        self.motion_runner = None
        self.metric_runner = None
        self.infiller_runner = None

        if self.gpu is not None and self.gpu != "":
            os.environ["CUDA_VISIBLE_DEVICES"] = str(self.gpu)

    def build_stage_args(self, video_path: str):
        class StageArgs:
            pass

        args = StageArgs()
        args.img_focal = self.img_focal
        args.video_path = video_path
        args.input_type = self.input_type
        args.checkpoint = self.checkpoint
        args.infiller_weight = self.infiller_weight
        args.chunk_batch_size = self.chunk_batch_size
        args.metric3d_batch_size = self.metric3d_batch_size
        args.detect_batch_size = self.detect_batch_size
        args.frame_backend = self.frame_backend
        args.vis_mode = "world"
        args.skip_vis = True
        return args

    def ensure_runner(self, stage: str):
        if stage == "detect_track" and self.detector_runner is None:
            from ultralytics import YOLO

            self.detector_runner = YOLO('./weights/external/detector.pt')

        if stage == "motion" and self.motion_runner is None:
            from scripts.scripts_test_video.hawor_video import build_motion_runner

            self.motion_runner = build_motion_runner(self.checkpoint)

        if stage == "slam" and self.metric_runner is None:
            from scripts.scripts_test_video.hawor_slam import build_metric3d_runner

            self.metric_runner = build_metric3d_runner()

        if stage == "infiller" and self.infiller_runner is None:
            from scripts.scripts_test_video.hawor_video import build_infiller_runner

            self.infiller_runner = build_infiller_runner(self.infiller_weight)



def build_stage_args(ns):
    class StageArgs:
        pass

    args = StageArgs()
    args.img_focal = ns.img_focal
    args.video_path = ns.video_path
    args.input_type = ns.input_type
    args.checkpoint = ns.checkpoint
    args.infiller_weight = ns.infiller_weight
    args.chunk_batch_size = ns.chunk_batch_size
    args.frame_backend = ns.frame_backend
    args.vis_mode = "world"
    args.skip_vis = True
    return args


def run_stage_with_runtime(runtime: WorkerRuntime, ns):
    stage_args = runtime.build_stage_args(ns.video_path)
    seq_folder = get_seq_folder(ns.video_path)

    if ns.resume and not ns.force and is_stage_complete(ns.stage, seq_folder, fast_check=True):
        return {
            "status": "skipped",
            "reason": "existing_valid_output",
        }

    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from scripts.scripts_test_video.hawor_video import run_infiller_for_video, run_motion_for_video

    runtime.ensure_runner(ns.stage)

    if ns.stage == "detect_track":
        start_idx, end_idx, _, _ = detect_track_video(
            stage_args,
            detector_runner=runtime.detector_runner,
            force=ns.force,
            detect_batch_size=ns.detect_batch_size,
        )
    else:
        start_idx, end_idx = get_track_range(seq_folder)

    if ns.stage == "motion":
        run_motion_for_video(
            stage_args,
            start_idx,
            end_idx,
            str(seq_folder),
            motion_runner=runtime.motion_runner,
        )
    elif ns.stage == "slam":
        hawor_slam(stage_args, start_idx, end_idx, metric_runner=runtime.metric_runner, metric3d_batch_size=ns.metric3d_batch_size)
    elif ns.stage == "infiller":
        tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
        frame_chunks_all = joblib.load(tracks_dir / "frame_chunks_all.npy")
        run_infiller_for_video(
            stage_args,
            start_idx,
            end_idx,
            frame_chunks_all,
            infiller_runner=runtime.infiller_runner,
        )
    elif ns.stage == "detect_track":
        pass
    else:
        raise ValueError(f"Unknown stage: {ns.stage}")

    validate_stage_output(ns.stage, seq_folder, start_idx, end_idx)

    # Create .done marker after successful validation
    done_marker = seq_folder / f".{ns.stage}.done"
    done_marker.touch()

    return {
        "status": "success",
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def worker_runtime_loop(ns):
    set_determinism(ns.seed)
    runtime = WorkerRuntime(
        gpu=ns.gpu,
        checkpoint=ns.checkpoint,
        infiller_weight=ns.infiller_weight,
        img_focal=ns.img_focal,
        input_type=ns.input_type,
        chunk_batch_size=ns.chunk_batch_size,
        frame_backend=ns.frame_backend,
    )

    with open(ns.video_list) as f:
        video_paths = [line.strip() for line in f if line.strip()]

    overall_success = True
    for video_path in video_paths:
        task_ns = argparse.Namespace(**vars(ns))
        task_ns.video_path = video_path

        common_fields = {
            "video": task_ns.video_path,
            "stage": task_ns.stage,
            "gpu": task_ns.gpu,
        }
        started_at = time.time()
        emit_event("stage_start", **common_fields)
        try:
            result = run_stage_with_runtime(runtime, task_ns)
            emit_event(
                "stage_end",
                **common_fields,
                status=result.get("status", "success"),
                elapsed_sec=round(time.time() - started_at, 3),
                reason=result.get("reason"),
                start_idx=result.get("start_idx"),
                end_idx=result.get("end_idx"),
            )
        except Exception as err:
            overall_success = False
            emit_event(
                "stage_end",
                **common_fields,
                status="failed",
                elapsed_sec=round(time.time() - started_at, 3),
                error=str(err),
            )
            traceback.print_exc()

    return overall_success


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

    if ns.resume and not ns.force and is_stage_complete(ns.stage, seq_folder, fast_check=True):
        return {
            "status": "skipped",
            "reason": "existing_valid_output",
        }

    from scripts.scripts_test_video.detect_track_video import detect_track_video
    from scripts.scripts_test_video.hawor_slam import hawor_slam
    from scripts.scripts_test_video.hawor_video import hawor_infiller, hawor_motion_estimation

    if ns.stage == "detect_track":
        start_idx, end_idx, _, _ = detect_track_video(stage_args, detect_batch_size=ns.detect_batch_size)
    else:
        start_idx, end_idx = get_track_range(seq_folder)

    if ns.stage == "motion":
        hawor_motion_estimation(stage_args, start_idx, end_idx, str(seq_folder))
    elif ns.stage == "slam":
        hawor_slam(stage_args, start_idx, end_idx, metric3d_batch_size=ns.metric3d_batch_size)
    elif ns.stage == "infiller":
        tracks_dir = seq_folder / f"tracks_{start_idx}_{end_idx}"
        frame_chunks_all = joblib.load(tracks_dir / "frame_chunks_all.npy")
        hawor_infiller(stage_args, start_idx, end_idx, frame_chunks_all)
    elif ns.stage == "detect_track":
        pass
    else:
        raise ValueError(f"Unknown stage: {ns.stage}")

    validate_stage_output(ns.stage, seq_folder, start_idx, end_idx)

    # Create .done marker after successful validation
    done_marker = seq_folder / f".{ns.stage}.done"
    done_marker.touch()

    return {
        "status": "success",
        "start_idx": start_idx,
        "end_idx": end_idx,
    }


def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", required=True, choices=STAGES)
    parser.add_argument("--video_path", type=str)
    parser.add_argument("--gpu", default="", type=str)
    parser.add_argument("--img_focal", type=float)
    parser.add_argument("--input_type", type=str, default="file")
    parser.add_argument("--checkpoint", type=str, default="./weights/hawor/checkpoints/hawor.ckpt")
    parser.add_argument("--infiller_weight", type=str, default="./weights/hawor/checkpoints/infiller.pt")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--chunk_batch_size", type=int, default=8)
    parser.add_argument("--metric3d_batch_size", type=int, default=8, help="Batch size for Metric3D depth estimation")
    # CRITICAL: detect_batch_size MUST be 1 for tracking to work correctly.
    # Tracking is stateful and sequential - each frame depends on previous frame's
    # Kalman filter state. Batching multiple frames breaks this dependency and
    # causes "LinAlgError: leading minor not positive definite" in tracker.
    #
    # For performance optimization, use cross-video batching (detect_video_batch_size)
    # instead, which processes multiple videos in parallel while maintaining
    # per-video tracker state.
    parser.add_argument("--detect_batch_size", type=int, default=1, help="MUST be 1 - frame-level batching breaks tracker state")
    parser.add_argument("--resume", dest="resume", action="store_true", default=True)
    parser.add_argument("--no-resume", dest="resume", action="store_false")
    parser.add_argument("--force", action="store_true", help="Ignore existing outputs and rerun this stage")
    parser.add_argument("--frame_backend", type=str, default="decord", choices=["decord", "opencv"], help="Frame decode backend")
    parser.add_argument("--video_list", type=str, help="Optional file with one video path per line for persistent worker mode")
    parser.add_argument("--persistent_worker", action="store_true", help="Run as long-lived stage worker for multiple videos")
    return parser


if __name__ == "__main__":
    args = get_parser().parse_args()

    # Validate detect_batch_size
    if args.detect_batch_size > 1:
        raise ValueError(
            f"detect_batch_size must be 1 (got {args.detect_batch_size}). "
            "Frame-level batching breaks YOLO tracker state and causes Kalman filter errors. "
            "Use --detect_video_batch_size to process multiple videos in parallel instead."
        )

    if args.persistent_worker:
        if not args.video_list:
            raise ValueError("--video_list is required when --persistent_worker is set")
        success = worker_runtime_loop(args)
        sys.exit(0 if success else 1)

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
