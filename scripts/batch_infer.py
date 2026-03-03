#!/usr/bin/env python3
"""
Multi-GPU batch inference scheduler for HaWoR.

Schedules multiple videos across GPU workers, executing stages sequentially
per video with retry logic, resume/skip support, and structured logging.
"""
import argparse
import json
import multiprocessing as mp
import os
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[1]
STAGES = ["detect_track", "motion", "slam", "infiller"]


class VideoTask:
    def __init__(self, video_path: str, run_id: str, log_dir: Path):
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        self.run_id = run_id
        self.log_dir = log_dir
        self.stage_status = {stage: "pending" for stage in STAGES}
        self.retry_count = {stage: 0 for stage in STAGES}
        self.start_time = None
        self.end_time = None

    def to_dict(self):
        return {
            "video_path": self.video_path,
            "video_name": self.video_name,
            "stage_status": self.stage_status,
            "retry_count": self.retry_count,
            "start_time": self.start_time,
            "end_time": self.end_time,
        }


class BatchScheduler:
    def __init__(
        self,
        video_paths: List[str],
        gpus: List[int],
        stages: List[str],
        max_retries: int,
        resume: bool,
        run_dir: Path,
        checkpoint: str,
        infiller_weight: str,
        img_focal: Optional[float],
    ):
        self.video_paths = video_paths
        self.gpus = gpus
        self.stages = stages
        self.max_retries = max_retries
        self.resume = resume
        self.run_dir = run_dir
        self.checkpoint = checkpoint
        self.infiller_weight = infiller_weight
        self.img_focal = img_focal

        self.log_dir = run_dir / "logs"
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.status_file = run_dir / "status.json"
        self.events_file = run_dir / "events.jsonl"

        self.tasks: Dict[str, VideoTask] = {}
        self.lock = mp.Lock()

        for vp in video_paths:
            task = VideoTask(vp, run_dir.name, self.log_dir)
            self.tasks[vp] = task

    def emit_event(self, event: str, **kwargs):
        payload = {
            "time": datetime.now(timezone.utc).isoformat(),
            "event": event,
            **kwargs,
        }
        with open(self.events_file, "a") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")

    def save_status(self):
        with self.lock:
            status_data = {
                "run_dir": str(self.run_dir),
                "gpus": self.gpus,
                "stages": self.stages,
                "tasks": {vp: task.to_dict() for vp, task in self.tasks.items()},
            }
            with open(self.status_file, "w") as f:
                json.dump(status_data, f, indent=2, ensure_ascii=False)

    def load_status(self):
        if not self.status_file.exists():
            return
        with open(self.status_file) as f:
            data = json.load(f)
        for vp, task_data in data.get("tasks", {}).items():
            if vp in self.tasks:
                self.tasks[vp].stage_status = task_data.get("stage_status", {})
                self.tasks[vp].retry_count = task_data.get("retry_count", {})
                self.tasks[vp].start_time = task_data.get("start_time")
                self.tasks[vp].end_time = task_data.get("end_time")

    def run_stage_subprocess(
        self, video_path: str, stage: str, gpu: int
    ) -> Tuple[int, str, str]:
        log_file = self.log_dir / f"{Path(video_path).stem}_{stage}.log"
        cmd = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "batch_worker.py"),
            "--stage", stage,
            "--video_path", video_path,
            "--gpu", str(gpu),
            "--checkpoint", self.checkpoint,
            "--infiller_weight", self.infiller_weight,
        ]
        if self.img_focal is not None:
            cmd.extend(["--img_focal", str(self.img_focal)])
        if self.resume:
            cmd.append("--resume")

        with open(log_file, "w") as f:
            proc = subprocess.Popen(
                cmd,
                stdout=f,
                stderr=subprocess.STDOUT,
                cwd=PROJECT_ROOT,
            )
            proc.wait()

        with open(log_file) as f:
            log_content = f.read()

        return proc.returncode, str(log_file), log_content

    def verify_stage_complete(self, video_path: str, stage: str) -> bool:
        """Verify that stage output actually exists on disk."""
        try:
            video_path_obj = Path(video_path)
            seq_folder = video_path_obj.parent / video_path_obj.stem

            # Import validation function from batch_worker
            sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
            from batch_worker import is_stage_complete

            return is_stage_complete(stage, seq_folder)
        except Exception:
            return False

    def process_video(self, video_path: str, gpu: int):
        task = self.tasks[video_path]
        task.start_time = datetime.now(timezone.utc).isoformat()
        self.emit_event("video_start", video=video_path, gpu=gpu)

        for stage in self.stages:
            # Check both status.json AND actual disk artifacts
            if task.stage_status[stage] == "completed":
                if self.verify_stage_complete(video_path, stage):
                    self.emit_event("stage_skip", video=video_path, stage=stage, gpu=gpu)
                    continue
                else:
                    # Status says completed but output missing - need to rerun
                    self.emit_event(
                        "stage_revalidate",
                        video=video_path,
                        stage=stage,
                        gpu=gpu,
                        reason="output_missing"
                    )
                    task.stage_status[stage] = "pending"

            success = False
            for attempt in range(self.max_retries + 1):
                task.retry_count[stage] = attempt
                task.stage_status[stage] = "running"
                self.save_status()

                self.emit_event(
                    "stage_attempt",
                    video=video_path,
                    stage=stage,
                    gpu=gpu,
                    attempt=attempt,
                )

                returncode, log_file, log_content = self.run_stage_subprocess(
                    video_path, stage, gpu
                )

                if returncode == 0:
                    task.stage_status[stage] = "completed"
                    self.save_status()
                    self.emit_event(
                        "stage_success",
                        video=video_path,
                        stage=stage,
                        gpu=gpu,
                        attempt=attempt,
                        log_file=log_file,
                    )
                    success = True
                    break
                else:
                    self.emit_event(
                        "stage_failure",
                        video=video_path,
                        stage=stage,
                        gpu=gpu,
                        attempt=attempt,
                        returncode=returncode,
                        log_file=log_file,
                    )

            if not success:
                task.stage_status[stage] = "failed"
                self.save_status()
                self.emit_event("video_failed", video=video_path, stage=stage, gpu=gpu)
                return False

        task.end_time = datetime.now(timezone.utc).isoformat()
        self.save_status()
        self.emit_event("video_completed", video=video_path, gpu=gpu)
        return True

    def worker_loop(self, gpu: int, video_queue: mp.Queue, result_queue: mp.Queue):
        while True:
            try:
                video_path = video_queue.get(timeout=1)
            except:
                break

            if video_path is None:
                break

            success = self.process_video(video_path, gpu)
            result_queue.put((video_path, success))

    def run(self):
        if self.resume:
            self.load_status()

        self.emit_event("batch_start", total_videos=len(self.video_paths), gpus=self.gpus)

        video_queue = mp.Queue()
        result_queue = mp.Queue()

        for vp in self.video_paths:
            video_queue.put(vp)

        for _ in self.gpus:
            video_queue.put(None)

        workers = []
        for gpu in self.gpus:
            p = mp.Process(target=self.worker_loop, args=(gpu, video_queue, result_queue))
            p.start()
            workers.append(p)

        for w in workers:
            w.join()

        results = []
        while not result_queue.empty():
            results.append(result_queue.get())

        success_count = sum(1 for _, success in results if success)
        fail_count = len(results) - success_count

        self.emit_event(
            "batch_end",
            total=len(results),
            success=success_count,
            failed=fail_count,
        )

        print(f"\n=== Batch Inference Complete ===")
        print(f"Total videos: {len(results)}")
        print(f"Success: {success_count}")
        print(f"Failed: {fail_count}")
        print(f"Run directory: {self.run_dir}")
        print(f"Status file: {self.status_file}")
        print(f"Events log: {self.events_file}")

        return fail_count == 0


def collect_videos(video_dir: Path, extensions=(".mp4", ".avi", ".mov")) -> List[str]:
    videos = []
    for ext in extensions:
        videos.extend(str(p) for p in video_dir.rglob(f"*{ext}"))
    return sorted(videos)


def get_parser():
    parser = argparse.ArgumentParser(
        description="Multi-GPU batch inference scheduler for HaWoR"
    )
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--video_list",
        type=str,
        help="Path to text file with one video path per line",
    )
    input_group.add_argument(
        "--video_dir",
        type=str,
        help="Directory to recursively search for video files",
    )

    parser.add_argument(
        "--gpus",
        type=str,
        default="0",
        help="Comma-separated GPU IDs (e.g., '0,1,2,3')",
    )
    parser.add_argument(
        "--stages",
        type=str,
        default="detect_track,motion,slam,infiller",
        help="Comma-separated stage names",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=2,
        help="Max retries per stage (default: 2)",
    )
    parser.add_argument(
        "--resume",
        dest="resume",
        action="store_true",
        default=True,
        help="Resume from existing outputs (default: True)",
    )
    parser.add_argument(
        "--no-resume",
        dest="resume",
        action="store_false",
        help="Ignore existing outputs and rerun all stages",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        help="Custom run directory (default: batch_runs/<timestamp>)",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="./weights/hawor/checkpoints/hawor.ckpt",
        help="Path to HaWoR checkpoint",
    )
    parser.add_argument(
        "--infiller_weight",
        type=str,
        default="./weights/hawor/checkpoints/infiller.pt",
        help="Path to infiller weights",
    )
    parser.add_argument(
        "--img_focal",
        type=float,
        help="Image focal length (optional)",
    )
    parser.add_argument(
        "--start",
        type=int,
        default=0,
        help="Start index of video list (inclusive, 0-based)",
    )
    parser.add_argument(
        "--end",
        type=int,
        default=None,
        help="End index of video list (exclusive, None means process all)",
    )

    return parser


def main():
    args = get_parser().parse_args()

    if args.video_list:
        with open(args.video_list) as f:
            video_paths = [line.strip() for line in f if line.strip()]
    else:
        video_paths = collect_videos(Path(args.video_dir))

    if not video_paths:
        print("Error: No videos found", file=sys.stderr)
        sys.exit(1)

    # Apply start-end slicing
    total_videos = len(video_paths)
    start_idx = args.start
    end_idx = args.end if args.end is not None else total_videos

    # Validate indices
    if start_idx < 0 or start_idx >= total_videos:
        print(f"Error: --start {start_idx} is out of range [0, {total_videos})", file=sys.stderr)
        sys.exit(1)
    if end_idx < start_idx or end_idx > total_videos:
        print(f"Error: --end {end_idx} is out of range [{start_idx}, {total_videos}]", file=sys.stderr)
        sys.exit(1)

    video_paths = video_paths[start_idx:end_idx]

    if not video_paths:
        print(f"Error: No videos in range [{start_idx}, {end_idx})", file=sys.stderr)
        sys.exit(1)

    gpus = [int(g.strip()) for g in args.gpus.split(",")]
    stages = [s.strip() for s in args.stages.split(",")]

    if args.run_dir:
        run_dir = Path(args.run_dir)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = PROJECT_ROOT / "batch_runs" / timestamp

    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Batch Inference Configuration ===")
    print(f"Total videos in list: {total_videos}")
    print(f"Processing range: [{start_idx}, {end_idx})")
    print(f"Videos to process: {len(video_paths)}")
    print(f"GPUs: {gpus}")
    print(f"Stages: {stages}")
    print(f"Max retries: {args.retries}")
    print(f"Resume: {args.resume}")
    print(f"Run directory: {run_dir}")
    print()

    scheduler = BatchScheduler(
        video_paths=video_paths,
        gpus=gpus,
        stages=stages,
        max_retries=args.retries,
        resume=args.resume,
        run_dir=run_dir,
        checkpoint=args.checkpoint,
        infiller_weight=args.infiller_weight,
        img_focal=args.img_focal,
    )

    success = scheduler.run()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

