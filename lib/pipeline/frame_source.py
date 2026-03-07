import cv2
import numpy as np


class BaseFrameSource:
    def __len__(self):
        raise NotImplementedError

    def get_frame(self, index: int, rgb: bool = False):
        raise NotImplementedError

    def iter_frames(self, rgb: bool = False):
        for idx in range(len(self)):
            yield idx, self.get_frame(idx, rgb=rgb)

    def get_size(self):
        frame = self.get_frame(0, rgb=False)
        h, w = frame.shape[:2]
        return h, w


class OpenCVVideoFrameSource(BaseFrameSource):
    def __init__(self, video_path: str):
        self.video_path = video_path
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {video_path}")
        self._num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        if self._num_frames <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

    def __len__(self):
        return self._num_frames

    def get_frame(self, index: int, rgb: bool = False):
        if index < 0 or index >= self._num_frames:
            raise IndexError(f"Frame index {index} out of range [0, {self._num_frames})")

        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
        ok, frame = cap.read()
        cap.release()
        if not ok or frame is None:
            raise RuntimeError(f"Failed to read frame {index} from {self.video_path}")

        if rgb:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame

    def iter_frames(self, rgb: bool = False):
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Failed to open video: {self.video_path}")

        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            yield idx, frame
            idx += 1
        cap.release()


class DecordVideoFrameSource(BaseFrameSource):
    def __init__(self, video_path: str):
        from decord import VideoReader, cpu

        self.video_path = video_path
        self._vr = VideoReader(video_path, ctx=cpu(0))
        self._num_frames = len(self._vr)
        if self._num_frames <= 0:
            raise RuntimeError(f"Video has no frames: {video_path}")

    def __len__(self):
        return self._num_frames

    def get_frame(self, index: int, rgb: bool = False):
        if index < 0 or index >= self._num_frames:
            raise IndexError(f"Frame index {index} out of range [0, {self._num_frames})")

        frame = self._vr[index].asnumpy()
        if rgb:
            return frame
        return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

    def iter_frames(self, rgb: bool = False):
        for idx in range(self._num_frames):
            frame = self._vr[idx].asnumpy()
            if not rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield idx, frame


class ImageFolderFrameSource(BaseFrameSource):
    def __init__(self, image_paths):
        self.image_paths = list(image_paths)
        if len(self.image_paths) == 0:
            raise RuntimeError("ImageFolderFrameSource requires non-empty image_paths")

    def __len__(self):
        return len(self.image_paths)

    def get_frame(self, index: int, rgb: bool = False):
        path = self.image_paths[index]
        frame = cv2.imread(path)
        if frame is None:
            raise RuntimeError(f"Failed to read image: {path}")
        if rgb:
            return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return frame


def build_frame_source(video_path: str, backend: str = "decord", fallback_backend: str = "opencv"):
    if backend == "decord":
        try:
            return DecordVideoFrameSource(video_path), "decord"
        except Exception:
            if fallback_backend == "opencv":
                return OpenCVVideoFrameSource(video_path), "opencv"
            raise

    if backend == "opencv":
        return OpenCVVideoFrameSource(video_path), "opencv"

    raise ValueError(f"Unsupported backend: {backend}")


def build_frame_source_auto(video_path: str, backend: str = "decord", fallback_backend: str = "opencv"):
    """
    Automatically choose frame source based on availability.

    Priority:
    1. If <video_dir>/<video_stem>/extracted_images/ exists, use ImageFolderFrameSource
    2. Otherwise, use video file with specified backend

    Returns:
        tuple: (frame_source, source_type)
            source_type: "extracted" or backend name ("decord"/"opencv")
    """
    from pathlib import Path
    import glob
    from natsort import natsorted

    video_path_obj = Path(video_path)
    video_dir = video_path_obj.parent
    video_stem = video_path_obj.stem

    # Check for extracted frames
    extracted_dir = video_dir / video_stem / "extracted_images"
    if extracted_dir.exists() and extracted_dir.is_dir():
        # Find all image files (try jpg first, then png)
        image_files = natsorted(glob.glob(str(extracted_dir / "*.jpg")))
        if not image_files:
            image_files = natsorted(glob.glob(str(extracted_dir / "*.png")))

        if image_files:
            print(f"✓ Using extracted frames from: {extracted_dir}")
            print(f"  Found {len(image_files)} frames")
            return ImageFolderFrameSource(image_files), "extracted"

    # Fallback to video file
    print(f"Using video file: {video_path}")
    return build_frame_source(video_path, backend=backend, fallback_backend=fallback_backend)
