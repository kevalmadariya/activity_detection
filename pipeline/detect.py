"""
pipeline.detect — Stage 2: Person Detection & Per-Person Clip Extraction.

Responsibilities:
    - Receives a ``CaptureResult`` (raw 4-second clip from Stage 1).
    - Runs YOLOv8 person detection on every frame.
    - Tracks each detected person across frames (ByteTrack).
    - For each tracked person, crops their bounding box (+ padding)
      from every frame and assembles it into a separate video clip.
    - Saves each person-clip as ``temp/person_<track_id>_<clip_id>.mp4``.
    - Returns a list of ``DetectionResult`` — one per detected person.

SOLID design:
    S — Only handles detection & crop extraction.
    O — Detector backend is swappable via ``PersonDetector`` interface.
    L — ``DetectionResult`` extends the pipeline contract cleanly.
    I — Callers only use ``extract_person_clips()``.
    D — Depends on abstract ``CaptureResult``, not ffmpeg/YOLO internals.
"""

from __future__ import annotations

import logging
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Output model for Stage 2
# ---------------------------------------------------------------------------

@dataclass
class PersonClip:
    """
    A single cropped video clip of one tracked person.

    Produced by Stage 2, consumed by Stage 3 (Vision LLM labelling).

    Attributes:
        clip_id:        Unique ID for this person-clip.
        person_id:      YOLO track ID within the source clip.
        source_clip_id: The ``CaptureResult.clip_id`` this came from.
        clip_path:      Saved ``.mp4`` path (cropped, person-only).
        frame_count:    Number of frames in the clip.
        fps:            Original frame rate.
        width:          Crop width (pixels).
        height:         Crop height (pixels).
        created_at:     Timestamp.
    """
    clip_id:        str
    person_id:      int
    source_clip_id: str
    clip_path:      Path
    frame_count:    int
    fps:            float
    width:          int
    height:         int
    created_at:     datetime = field(default_factory=datetime.now)

    def __repr__(self) -> str:
        return (
            f"PersonClip(person_id={self.person_id}, "
            f"frames={self.frame_count}, "
            f"path={self.clip_path.name})"
        )


@dataclass
class DetectionResult:
    """
    Full output of the detection stage for one source clip.

    Attributes:
        source_clip_id: Links back to ``CaptureResult.clip_id``.
        source_path:    Path of the raw clip that was processed.
        persons:        All person-clips extracted from this raw clip.
        total_persons:  Number of unique persons detected.
        error:          Set if the whole clip failed to process.
    """
    source_clip_id: str
    source_path:    Path
    persons:        List[PersonClip] = field(default_factory=list)
    error:          Optional[str]    = None

    @property
    def total_persons(self) -> int:
        return len(self.persons)

    @property
    def is_success(self) -> bool:
        return self.error is None

    def __repr__(self) -> str:
        tag = "✅" if self.is_success else "❌"
        return (
            f"{tag} DetectionResult("
            f"persons={self.total_persons}, "
            f"source={self.source_path.name})"
        )


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

_CROP_PADDING = 0.20        # 20% padding around bounding box
_OUTPUT_SIZE  = (312, 312)  # Resize each crop to match X3D input


def _expand_bbox(
    x1: int, y1: int, x2: int, y2: int,
    frame_w: int, frame_h: int,
    padding: float = _CROP_PADDING,
) -> Tuple[int, int, int, int]:
    """Expand bounding box by ``padding`` fraction, clamped to frame."""
    pad_x = int((x2 - x1) * padding)
    pad_y = int((y2 - y1) * padding)
    return (
        max(0, x1 - pad_x),
        max(0, y1 - pad_y),
        min(frame_w, x2 + pad_x),
        min(frame_h, y2 + pad_y),
    )


def _crop_frame(
    frame: np.ndarray,
    x1: int, y1: int, x2: int, y2: int,
) -> np.ndarray:
    """Crop and resize a single frame to ``_OUTPUT_SIZE``."""
    crop = frame[y1:y2, x1:x2]
    if crop.size == 0:
        return np.zeros((*_OUTPUT_SIZE[::-1], 3), dtype=np.uint8)
    return cv2.resize(crop, _OUTPUT_SIZE)


def _save_person_clip(
    frames: List[np.ndarray],
    output_path: Path,
    fps: float,
) -> bool:
    """Write a list of BGR frames as an mp4 clip. Returns True on success."""
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h   = _OUTPUT_SIZE
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

    if not writer.isOpened():
        logger.error("VideoWriter could not open: %s", output_path)
        return False

    for f in frames:
        writer.write(f)

    writer.release()

    if not output_path.exists() or output_path.stat().st_size == 0:
        logger.error("Saved clip is empty: %s", output_path)
        return False

    return True


# ---------------------------------------------------------------------------
# Core public function
# ---------------------------------------------------------------------------

def extract_person_clips(
    source_clip_path: Path,
    source_clip_id:   str,
    output_dir:       Path,
    *,
    model_size:       str   = "n",          # yolov8 variant: n/s/m/l/x
    conf_threshold:   float = 0.40,
    min_frames:       int   = 30,           # discard persons seen < N frames
    padding:          float = _CROP_PADDING,
) -> DetectionResult:
    """
    Detect all persons in a clip and extract one cropped clip per person.

    For each unique tracked person whose bounding box is visible for at
    least ``min_frames`` frames, this function:
        1. Crops that person from every frame (with padding).
        2. Resizes to 312×312 (X3D input size).
        3. Saves as ``<output_dir>/person_<track_id>_<clip_id>.mp4``.

    Args:
        source_clip_path: Path to raw ``.mp4`` from Stage 1.
        source_clip_id:   ``CaptureResult.clip_id`` for traceability.
        output_dir:       Where to write person clips.
        model_size:       YOLOv8 nano (``"n"``) is fastest; ``"s"`` is better.
        conf_threshold:   Minimum detection confidence (0–1).
        min_frames:       Minimum frames a person must appear to be saved.
        padding:          Fractional padding around bounding box (default 20%).

    Returns:
        :class:`DetectionResult` containing one :class:`PersonClip` per person.
    """
    # Late import so the module loads even before YOLO is installed
    try:
        from ultralytics import YOLO
    except ImportError:
        return DetectionResult(
            source_clip_id=source_clip_id,
            source_path=source_clip_path,
            error="ultralytics not installed — run: uv add ultralytics",
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # 1. Open video
    # ------------------------------------------------------------------
    cap = cv2.VideoCapture(str(source_clip_path))
    if not cap.isOpened():
        return DetectionResult(
            source_clip_id=source_clip_id,
            source_path=source_clip_path,
            error=f"Cannot open video: {source_clip_path}",
        )

    fps     = cap.get(cv2.CAP_PROP_FPS) or 25.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Read all frames into memory (4s @ 25fps = 100 frames — manageable)
    raw_frames: List[np.ndarray] = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        raw_frames.append(frame)
    cap.release()

    if not raw_frames:
        return DetectionResult(
            source_clip_id=source_clip_id,
            source_path=source_clip_path,
            error="Video has no readable frames",
        )

    logger.info(
        "Loaded %d frames (%.1f fps) from %s",
        len(raw_frames), fps, source_clip_path.name,
    )

    # ------------------------------------------------------------------
    # 2. Run YOLO tracking on all frames
    # ------------------------------------------------------------------
    model_name = f"yolov8{model_size}.pt"
    logger.info("Loading detector: %s", model_name)
    model = YOLO(model_name)

    # track() returns one Results object per frame
    tracking_results = model.track(
        source=str(source_clip_path),
        classes=[0],                    # 0 = person
        conf=conf_threshold,
        persist=True,
        tracker="bytetrack.yaml",
        verbose=False,
        stream=True,                    # memory-efficient
    )

    # ------------------------------------------------------------------
    # 3. Collect per-track bounding boxes (frame_index → bbox)
    # ------------------------------------------------------------------
    # Structure: {track_id: {frame_idx: (x1,y1,x2,y2)}}
    tracks: Dict[int, Dict[int, Tuple[int, int, int, int]]] = {}

    for frame_idx, result in enumerate(tracking_results):
        if result.boxes is None:
            continue
        for box in result.boxes:
            if box.id is None:
                continue
            track_id = int(box.id.item())
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)

            if track_id not in tracks:
                tracks[track_id] = {}
            tracks[track_id][frame_idx] = (x1, y1, x2, y2)

    logger.info("Detected %d unique person tracks", len(tracks))

    # ------------------------------------------------------------------
    # 4. Build a cropped clip for each person with enough frames
    # ------------------------------------------------------------------
    person_clips: List[PersonClip] = []

    for track_id, frame_bboxes in tracks.items():
        if len(frame_bboxes) < min_frames:
            logger.debug(
                "Track %d skipped — only %d frames (min=%d)",
                track_id, len(frame_bboxes), min_frames,
            )
            continue

        # Crop person from each frame where they were detected,
        # using the *nearest known bbox* for frames without a detection
        sorted_frame_idxs = sorted(frame_bboxes.keys())
        cropped_frames: List[np.ndarray] = []

        last_bbox = frame_bboxes[sorted_frame_idxs[0]]

        for frame_idx in range(len(raw_frames)):
            # Use detected bbox if available, else carry forward last known
            bbox = frame_bboxes.get(frame_idx, last_bbox)
            if frame_idx in frame_bboxes:
                last_bbox = bbox

            x1, y1, x2, y2 = _expand_bbox(
                *bbox, frame_w, frame_h, padding
            )
            cropped = _crop_frame(raw_frames[frame_idx], x1, y1, x2, y2)
            cropped_frames.append(cropped)

        # ------------------------------------------------------------------
        # 5. Save clip
        # ------------------------------------------------------------------
        clip_id   = uuid.uuid4().hex[:8]
        filename  = f"person_{track_id}_{source_clip_id[:8]}_{clip_id}.mp4"
        clip_path = output_dir / filename

        ok = _save_person_clip(cropped_frames, clip_path, fps)
        if not ok:
            logger.warning("Failed to save clip for track %d", track_id)
            continue

        person_clip = PersonClip(
            clip_id        = clip_id,
            person_id      = track_id,
            source_clip_id = source_clip_id,
            clip_path      = clip_path,
            frame_count    = len(cropped_frames),
            fps            = fps,
            width          = _OUTPUT_SIZE[0],
            height         = _OUTPUT_SIZE[1],
        )
        person_clips.append(person_clip)

        logger.info(
            "Saved person %d → %s (%d frames, %.1f KB)",
            track_id,
            clip_path.name,
            len(cropped_frames),
            clip_path.stat().st_size / 1024,
        )

    return DetectionResult(
        source_clip_id = source_clip_id,
        source_path    = source_clip_path,
        persons        = person_clips,
    )


# ---------------------------------------------------------------------------
# Batch helper — processes a list of CaptureResults directly
# ---------------------------------------------------------------------------

def extract_persons_from_captures(
    capture_results,       # List[CaptureResult] — avoid circular import
    output_dir: Path,
    **kwargs,
) -> List[DetectionResult]:
    """
    Run person detection on every successful clip from Stage 1.

    Args:
        capture_results: List of ``CaptureResult`` from ``capture_clips()``.
        output_dir:      Root folder for all person-clip outputs.
        **kwargs:        Forwarded to ``extract_person_clips()``.

    Returns:
        One ``DetectionResult`` per successful capture.
    """
    results = []
    successful = [r for r in capture_results if r.is_success and r.clip_path]
    logger.info(
        "Running detection on %d / %d clips",
        len(successful), len(capture_results),
    )

    for i, cap_result in enumerate(successful, 1):
        logger.info(
            "Processing clip %d/%d: %s",
            i, len(successful), cap_result.clip_path.name,
        )
        det = extract_person_clips(
            source_clip_path = cap_result.clip_path,
            source_clip_id   = cap_result.clip_id,
            output_dir       = output_dir,
            **kwargs,
        )
        results.append(det)

    return results
