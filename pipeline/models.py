"""
pipeline.models — Shared data models for all pipeline stages.

Each pipeline component produces and consumes these models, enabling
loose coupling between stages (Dependency Inversion Principle).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class StreamMode(Enum):
    """RTSP stream mode — playback (history) or live."""
    PLAYBACK = "playback"
    LIVE     = "realmonitor"


class CaptureStatus(Enum):
    """Outcome of a single clip capture attempt."""
    SUCCESS = "success"
    FAILED  = "failed"
    TIMEOUT = "timeout"


# ---------------------------------------------------------------------------
# Camera Configuration
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class CameraConfig:
    """
    Immutable camera connection descriptor.

    Attributes:
        camera_id:  Logical name (e.g. ``"CAMERA_URL_1"``).
        base_url:   Host:port loaded from ``.env`` (no scheme, no creds).
        username:   RTSP auth user.
        password:   RTSP auth password (URL-encoded if needed).
    """
    camera_id: str
    base_url:  str
    username:  str
    password:  str


# ---------------------------------------------------------------------------
# Capture Request / Result
# ---------------------------------------------------------------------------

@dataclass
class CaptureRequest:
    """
    Describes *what* to capture from a camera.

    Attributes:
        camera:       Which camera to connect to.
        channel:      DVR/NVR channel number.
        subtype:      Stream subtype (0 = main, 1 = sub).
        mode:         ``PLAYBACK`` for history, ``LIVE`` for realtime.
        start_time:   For playback — the starting timestamp to replay from.
                      Format: ``"YYYY_MM_DD_HH_MM_SS"``.
                      Ignored when ``mode`` is ``LIVE``.
        duration_sec: How many seconds to record.
        output_dir:   Directory where the captured clip will be saved.
    """
    camera:       CameraConfig
    channel:      int
    subtype:      int            = 0
    mode:         StreamMode     = StreamMode.PLAYBACK
    start_time:   Optional[str]  = None       # "2026_06_02_12_00_00"
    duration_sec: float          = 4.0
    output_dir:   Path           = field(default_factory=lambda: Path("temp"))

    def __post_init__(self) -> None:
        if self.mode is StreamMode.PLAYBACK and not self.start_time:
            raise ValueError(
                "start_time is required when mode is PLAYBACK.  "
                "Pass a string like '2026_06_02_12_00_00'."
            )
        # Ensure output_dir is a Path
        self.output_dir = Path(self.output_dir)


@dataclass
class CaptureResult:
    """
    Outcome produced by the capture stage.

    The next pipeline stage (person detection) consumes this object directly.

    Attributes:
        clip_id:      Unique identifier for this clip.
        clip_path:    Absolute path to the saved ``.mp4`` file.
        camera_id:    Which camera produced this clip.
        channel:      Channel that was recorded.
        mode:         Live or playback.
        start_time:   Playback start time (if applicable).
        duration_sec: Requested duration.
        status:       Whether the capture succeeded.
        error:        Error message when ``status`` is not ``SUCCESS``.
        captured_at:  When this capture was executed.
    """
    clip_id:      str
    clip_path:    Optional[Path]
    camera_id:    str
    channel:      int
    mode:         StreamMode
    start_time:   Optional[str]
    duration_sec: float
    status:       CaptureStatus
    error:        Optional[str]  = None
    captured_at:  datetime       = field(default_factory=datetime.now)

    @property
    def is_success(self) -> bool:
        return self.status is CaptureStatus.SUCCESS

    def __repr__(self) -> str:
        tag = "✅" if self.is_success else "❌"
        return (
            f"{tag} CaptureResult(clip_id={self.clip_id!r}, "
            f"status={self.status.value}, path={self.clip_path})"
        )
