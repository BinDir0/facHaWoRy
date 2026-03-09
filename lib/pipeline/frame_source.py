import cv2
import numpy as np
import os
import queue
import threading

# Try to import turbojpeg for faster JPEG decoding
try:
    from turbojpeg import TurboJPEG
    TURBOJPEG_AVAILABLE = True
except ImportError:
    TURBOJPEG_AVAILABLE = False


class PrefetchingFrameIterator:
    """Prefetch frames in background thread to hide I/O latency."""

    def __init__(self, frame_source, rgb=False, buffer_size=16):
        self.frame_source = frame_source
        self.rgb = rgb
        self.buffer_size = buffer_size
        self.queue = queue.Queue(maxsize=buffer_size)
        self.stop_event = threading.Event()
        self.exception = None

        self.thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.thread.start()

    def _prefetch_worker(self):
        try:
            for idx, frame in self.frame_source.iter_frames(rgb=self.rgb):
                if self.stop_event.is_set():
                    break
                self.queue.put((idx, frame))
            self.queue.put(None)  # Sentinel
        except Exception as e:
            self.exception = e
            self.queue.put(None)

    def __iter__(self):
        return self

    def __next__(self):
        if self.exception:
            raise self.exception
        item = self.queue.get()
        if item is None:
            raise StopIteration
        return item

    def __del__(self):
        self.stop_event.set()
        try:
            while not self.queue.empty():
                self.queue.get_nowait()
        except:
            pass


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
    def __init__(self, video_path: str, use_gpu: bool = True):
        from decord import VideoReader, cpu, gpu

        self.video_path = video_path
        # Use GPU decoding if available (much faster than CPU)
        if use_gpu:
            try:
                self._vr = VideoReader(video_path, ctx=gpu(0))
                self.use_gpu = True
            except Exception:
                # Fallback to CPU if GPU not available
                self._vr = VideoReader(video_path, ctx=cpu(0))
                self.use_gpu = False
        else:
            self._vr = VideoReader(video_path, ctx=cpu(0))
            self.use_gpu = False

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

    def get_batch(self, indices, rgb: bool = False):
        """Batch frame extraction - much faster than individual get_frame calls"""
        frames = self._vr.get_batch(indices).asnumpy()
        if not rgb:
            # Convert RGB to BGR for all frames
            frames = np.stack([cv2.cvtColor(f, cv2.COLOR_RGB2BGR) for f in frames])
        return frames

    def iter_frames(self, rgb: bool = False):
        for idx in range(self._num_frames):
            frame = self._vr[idx].asnumpy()
            if not rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            yield idx, frame


class ImageFolderFrameSource(BaseFrameSource):
    def __init__(self, image_paths, use_turbojpeg=True):
        self.image_paths = list(image_paths)

        # DEBUG logging
        QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"
        if not QUIET_MODE:
            print(f"DEBUG: ImageFolderFrameSource initialized with {len(self.image_paths)} frames")
            if len(self.image_paths) > 0:
                print(f"DEBUG: First frame: {self.image_paths[0]}")
                print(f"DEBUG: Last frame: {self.image_paths[-1]}")

        if len(self.image_paths) == 0:
            raise RuntimeError("ImageFolderFrameSource requires non-empty image_paths")

        # Initialize turbojpeg decoder if available and requested
        self.use_turbojpeg = use_turbojpeg and TURBOJPEG_AVAILABLE
        if self.use_turbojpeg:
            self.jpeg_decoder = TurboJPEG()
        else:
            self.jpeg_decoder = None

    def __len__(self):
        return len(self.image_paths)

    def get_frame(self, index: int, rgb: bool = False):
        # Bounds check with helpful error message
        if index < 0 or index >= len(self.image_paths):
            raise IndexError(
                f"Frame index {index} out of range [0, {len(self.image_paths)}). "
                f"This usually means track data references frames that don't exist in extracted frames. "
                f"Total frames available: {len(self.image_paths)}"
            )

        path = self.image_paths[index]

        # Use turbojpeg for JPEG files if available (2-3x faster than cv2.imread)
        if self.use_turbojpeg and path.lower().endswith(('.jpg', '.jpeg')):
            try:
                with open(path, 'rb') as f:
                    jpeg_data = f.read()
                # Decode directly to RGB or BGR
                if rgb:
                    frame = self.jpeg_decoder.decode(jpeg_data, pixel_format=0)  # RGB
                else:
                    frame = self.jpeg_decoder.decode(jpeg_data, pixel_format=1)  # BGR
                return frame
            except Exception:
                # Fallback to cv2 if turbojpeg fails
                pass

        # Fallback to cv2.imread
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
    import os
    from natsort import natsorted

    # Check if we should suppress verbose output
    QUIET_MODE = os.environ.get("HAWOR_QUIET", "0") == "1"

    video_path_obj = Path(video_path)
    video_dir = video_path_obj.parent
    video_stem = video_path_obj.stem

    # Check for extracted frames
    extracted_dir = video_dir / video_stem / "extracted_images"

    # DEBUG logging
    if not QUIET_MODE:
        print(f"DEBUG: build_frame_source_auto() called with video_path={video_path}")
        print(f"DEBUG: video_dir={video_dir}, video_stem={video_stem}")
        print(f"DEBUG: Checking for extracted frames at: {extracted_dir}")
        print(f"DEBUG: extracted_dir.exists()={extracted_dir.exists()}, is_dir()={extracted_dir.is_dir() if extracted_dir.exists() else 'N/A'}")

    if extracted_dir.exists() and extracted_dir.is_dir():
        # Find all image files (try jpg first, then png)
        image_files = natsorted(glob.glob(str(extracted_dir / "*.jpg")))
        if not QUIET_MODE:
            print(f"DEBUG: Found {len(image_files)} .jpg files")

        if not image_files:
            image_files = natsorted(glob.glob(str(extracted_dir / "*.png")))
            if not QUIET_MODE:
                print(f"DEBUG: Found {len(image_files)} .png files")

        if image_files:
            if not QUIET_MODE:
                print(f"✓ Using extracted frames from: {extracted_dir}")
                print(f"  Found {len(image_files)} frames")
            return ImageFolderFrameSource(image_files), "extracted"
        else:
            if not QUIET_MODE:
                print(f"WARNING: extracted_images directory exists but no image files found!")

    # Fallback to video file
    if not QUIET_MODE:
        print(f"Using video file: {video_path}")
    return build_frame_source(video_path, backend=backend, fallback_backend=fallback_backend)
