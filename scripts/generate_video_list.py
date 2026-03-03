#!/usr/bin/env python3
"""
Generate video list from BuildAI-processed dataset structure.

The dataset is organized as:
/share_data/lvjianan/datasets/BuildAI-processed/factory_{id}/worker_{id}/processed/factory{id}_worker{id}_{id}_crop{id}.mp4

This script recursively finds all videos matching this pattern and outputs a list.
"""

import argparse
import sys
from pathlib import Path


def find_videos(base_dir, pattern="**/*.mp4", sort=True):
    """
    Find all video files in the base directory.

    Args:
        base_dir: Base directory to search
        pattern: Glob pattern for video files
        sort: Whether to sort the results

    Returns:
        List of video file paths
    """
    base_path = Path(base_dir)

    if not base_path.exists():
        print(f"Error: Directory does not exist: {base_dir}", file=sys.stderr)
        sys.exit(1)

    if not base_path.is_dir():
        print(f"Error: Not a directory: {base_dir}", file=sys.stderr)
        sys.exit(1)

    print(f"Searching for videos in: {base_dir}", file=sys.stderr)
    print(f"Pattern: {pattern}", file=sys.stderr)

    videos = list(base_path.glob(pattern))

    if sort:
        videos = sorted(videos)

    print(f"Found {len(videos)} videos", file=sys.stderr)

    return [str(v.resolve()) for v in videos]


def filter_by_factory(videos, factory_ids):
    """Filter videos by factory IDs."""
    if not factory_ids:
        return videos

    factory_set = set(factory_ids)
    filtered = []

    for video in videos:
        # Extract factory ID from path
        # Example: .../factory_1/worker_2/...
        parts = Path(video).parts
        for part in parts:
            if part.startswith("factory_"):
                try:
                    fid = int(part.split("_")[1])
                    if fid in factory_set:
                        filtered.append(video)
                        break
                except (IndexError, ValueError):
                    continue

    return filtered


def filter_by_worker(videos, worker_ids):
    """Filter videos by worker IDs."""
    if not worker_ids:
        return videos

    worker_set = set(worker_ids)
    filtered = []

    for video in videos:
        # Extract worker ID from path
        # Example: .../worker_2/processed/...
        parts = Path(video).parts
        for part in parts:
            if part.startswith("worker_"):
                try:
                    wid = int(part.split("_")[1])
                    if wid in worker_set:
                        filtered.append(video)
                        break
                except (IndexError, ValueError):
                    continue

    return filtered


def main():
    parser = argparse.ArgumentParser(
        description="Generate video list from BuildAI-processed dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Generate list of all videos
  python scripts/generate_video_list.py \\
      --base_dir /share_data/lvjianan/datasets/BuildAI-processed \\
      --output videos.txt

  # Filter by specific factories
  python scripts/generate_video_list.py \\
      --base_dir /share_data/lvjianan/datasets/BuildAI-processed \\
      --factory 1 2 3 \\
      --output factory_1_2_3.txt

  # Filter by specific workers
  python scripts/generate_video_list.py \\
      --base_dir /share_data/lvjianan/datasets/BuildAI-processed \\
      --worker 1 2 \\
      --output worker_1_2.txt

  # Combine filters
  python scripts/generate_video_list.py \\
      --base_dir /share_data/lvjianan/datasets/BuildAI-processed \\
      --factory 1 \\
      --worker 1 2 3 \\
      --output factory1_workers123.txt

  # Output to stdout (for piping)
  python scripts/generate_video_list.py \\
      --base_dir /share_data/lvjianan/datasets/BuildAI-processed \\
      --no-output
        """
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Base directory of BuildAI-processed dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file path (default: print to stdout)"
    )
    parser.add_argument(
        "--no-output",
        action="store_true",
        help="Print to stdout instead of file"
    )
    parser.add_argument(
        "--pattern",
        type=str,
        default="**/*.mp4",
        help="Glob pattern for video files (default: **/*.mp4)"
    )
    parser.add_argument(
        "--factory",
        type=int,
        nargs="+",
        help="Filter by factory IDs (e.g., --factory 1 2 3)"
    )
    parser.add_argument(
        "--worker",
        type=int,
        nargs="+",
        help="Filter by worker IDs (e.g., --worker 1 2)"
    )
    parser.add_argument(
        "--no-sort",
        action="store_true",
        help="Do not sort the video list"
    )

    args = parser.parse_args()

    # Find all videos
    videos = find_videos(args.base_dir, args.pattern, sort=not args.no_sort)

    if not videos:
        print("No videos found!", file=sys.stderr)
        sys.exit(1)

    # Apply filters
    if args.factory:
        print(f"Filtering by factories: {args.factory}", file=sys.stderr)
        videos = filter_by_factory(videos, args.factory)
        print(f"After factory filter: {len(videos)} videos", file=sys.stderr)

    if args.worker:
        print(f"Filtering by workers: {args.worker}", file=sys.stderr)
        videos = filter_by_worker(videos, args.worker)
        print(f"After worker filter: {len(videos)} videos", file=sys.stderr)

    if not videos:
        print("No videos match the filters!", file=sys.stderr)
        sys.exit(1)

    # Output
    if args.no_output or not args.output:
        # Print to stdout
        for video in videos:
            print(video)
    else:
        # Write to file
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for video in videos:
                f.write(f"{video}\n")

        print(f"Video list written to: {args.output}", file=sys.stderr)
        print(f"Total videos: {len(videos)}", file=sys.stderr)


if __name__ == "__main__":
    main()
