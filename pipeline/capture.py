"""
pipeline.capture — RTSP clip capture using ffmpeg.

This is **Stage 1** of the activity-detection pipeline.
It connects to an RTSP camera (live or playback/history), records a clip
of the requested duration, and returns a :class:`CaptureResult` that the
next stage (person detection / tracking) can consume directly.

Design principles applied:
    S — Single Responsibility: Only handles clip capture.
    O — Open/Closed: New stream modes or transports are added via
        ``StreamMode`` enum + ``_build_rtsp_url()`` without touching
        existing capture logic.
    L — Liskov: ``CaptureResult`` is consumed uniformly regardless
        of whether the source was live or playback.
    I — Callers only depend on ``capture_clip()`` / ``capture_clips()``.
    D — Depends on abstract models (``CaptureRequest`` / ``CaptureResult``),
        not on ffmpeg internals.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Sequence

import cv2

from pipeline.models import (
    CameraConfig,
    CaptureRequest,
    CaptureResult,
    CaptureStatus,
    StreamMode,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# URL builder (internal)
# ---------------------------------------------------------------------------

def _build_rtsp_url(request: CaptureRequest) -> str:
    """
    Assemble a full RTSP URL from a :class:`CaptureRequest`.

    Playback example:
        rtsp://user:pass@host:port/cam/playback?channel=78&subtype=0&starttime=2026_06_02_12_00_00

    Live example:
        rtsp://user:pass@host:port/cam/realmonitor?channel=78&subtype=0
    """
    cam  = request.camera
    cred = f"{cam.username}:{cam.password}@" if cam.username else ""
    path = f"/cam/{request.mode.value}"

    params = f"channel={request.channel}&subtype={request.subtype}"
    if request.mode is StreamMode.PLAYBACK and request.start_time:
        params += f"&starttime={request.start_time}"

    return f"rtsp://{cred}{cam.base_url}{path}?{params}"


# ---------------------------------------------------------------------------
# Filename generator (internal)
# ---------------------------------------------------------------------------

def _generate_clip_filename(request: CaptureRequest, clip_id: str) -> str:
    """
    Produce a descriptive, collision-free filename.

    Format:  ``ch{channel}_{mode}_{timestamp}_{short_id}.mp4``
    Example: ``ch78_playback_20260602_120000_a1b2c3d4.mp4``
    """
    ts = (
        request.start_time.replace("_", "")[:14]     # compact playback ts
        if request.start_time
        else datetime.now().strftime("%Y%m%d_%H%M%S") # fallback
    )
    short_id = clip_id[:8]
    mode_tag = request.mode.value
    return f"ch{request.channel}_{mode_tag}_{ts}_{short_id}.mp4"


# ---------------------------------------------------------------------------
# Core capture function
# ---------------------------------------------------------------------------

def capture_clip(request: CaptureRequest) -> CaptureResult:
    """
    Capture a single clip from an RTSP stream and save it to disk using OpenCV.

    This function:
        1. Builds the RTSP URL from the request.
        2. Ensures the output directory exists.
        3. Opens the stream using OpenCV and records the clip.
        4. Returns a :class:`CaptureResult` for the next pipeline stage.

    Args:
        request: A fully populated :class:`CaptureRequest`.

    Returns:
        :class:`CaptureResult` with ``status=SUCCESS`` on success, or
        ``FAILED`` / ``TIMEOUT`` with an error message on failure.
    """
    clip_id  = uuid.uuid4().hex
    rtsp_url = _build_rtsp_url(request)

    # Prepare output path
    request.output_dir.mkdir(parents=True, exist_ok=True)
    filename  = _generate_clip_filename(request, clip_id)
    clip_path = (request.output_dir / filename).resolve()

    # Mask credentials in logs
    safe_url = rtsp_url
    if request.camera.username and request.camera.password:
        safe_url = rtsp_url.replace(
            f"{request.camera.username}:{request.camera.password}@", "***:***@"
        )
    logger.info(
        "Capturing %.1fs clip  camera=%s  channel=%d  mode=%s  url=%s (using OpenCV)",
        request.duration_sec,
        request.camera.camera_id,
        request.channel,
        request.mode.value,
        safe_url,
    )

    # Set RTSP transport to TCP via env variable to prevent packet loss
    os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;tcp"

    cap = cv2.VideoCapture(rtsp_url)
    if not cap.isOpened():
        err_msg = "OpenCV failed to open RTSP stream"
        logger.error("Capture FAILED: %s", err_msg)
        return CaptureResult(
            clip_id=clip_id,
            clip_path=None,
            camera_id=request.camera.camera_id,
            channel=request.channel,
            mode=request.mode,
            start_time=request.start_time,
            duration_sec=request.duration_sec,
            status=CaptureStatus.FAILED,
            error=err_msg,
        )

    # Read stream parameters
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps <= 0 or fps > 100:
        fps = 25.0

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if not width or not height or width <= 0 or height <= 0:
        width, height = 1920, 1080

    # Define the codec and create VideoWriter object.
    # mp4v is standard for .mp4 container
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(clip_path), fourcc, fps, (width, height))

    if not out.isOpened():
        cap.release()
        err_msg = "OpenCV failed to open VideoWriter"
        logger.error("Capture FAILED: %s", err_msg)
        return CaptureResult(
            clip_id=clip_id,
            clip_path=None,
            camera_id=request.camera.camera_id,
            channel=request.channel,
            mode=request.mode,
            start_time=request.start_time,
            duration_sec=request.duration_sec,
            status=CaptureStatus.FAILED,
            error=err_msg,
        )

    # Start capturing frames
    start_time = time.time()
    frames_written = 0
    # Add a safety margin to the timeout to avoid infinite hang
    timeout_duration = request.duration_sec + 10.0

    try:
        while (time.time() - start_time) < request.duration_sec:
            if (time.time() - start_time) > timeout_duration:
                logger.warning("Capture timed out (safety margin reached)")
                break

            ret, frame = cap.read()
            if not ret:
                logger.warning("Capture ended: no more frames or stream dropped")
                break

            out.write(frame)
            frames_written += 1
    except Exception as e:
        logger.exception("Error during OpenCV frame capture")
        cap.release()
        out.release()
        return CaptureResult(
            clip_id=clip_id,
            clip_path=None,
            camera_id=request.camera.camera_id,
            channel=request.channel,
            mode=request.mode,
            start_time=request.start_time,
            duration_sec=request.duration_sec,
            status=CaptureStatus.FAILED,
            error=str(e),
        )

    cap.release()
    out.release()

    # Verify file was actually created and is non-empty
    if not clip_path.exists() or clip_path.stat().st_size == 0:
        logger.error("Capture produced empty file: %s", clip_path)
        return CaptureResult(
            clip_id=clip_id,
            clip_path=None,
            camera_id=request.camera.camera_id,
            channel=request.channel,
            mode=request.mode,
            start_time=request.start_time,
            duration_sec=request.duration_sec,
            status=CaptureStatus.FAILED,
            error="OpenCV completed but output file is empty or missing",
        )

    logger.info(
        "Capture OK → %s (%.1f KB, %d frames)",
        clip_path,
        clip_path.stat().st_size / 1024,
        frames_written,
    )

    return CaptureResult(
        clip_id=clip_id,
        clip_path=clip_path,
        camera_id=request.camera.camera_id,
        channel=request.channel,
        mode=request.mode,
        start_time=request.start_time,
        duration_sec=request.duration_sec,
        status=CaptureStatus.SUCCESS,
    )


# ---------------------------------------------------------------------------
# Batch capture (convenience)
# ---------------------------------------------------------------------------

def capture_clips(requests: Sequence[CaptureRequest]) -> List[CaptureResult]:
    """
    Capture multiple clips sequentially.

    Args:
        requests: An iterable of :class:`CaptureRequest` objects.

    Returns:
        A list of :class:`CaptureResult` objects — one per request,
        in the same order.
    """
    results = []
    for i, req in enumerate(requests, 1):
        logger.info("Capturing clip %d / %d …", i, len(requests))
        results.append(capture_clip(req))
    return results


# ---------------------------------------------------------------------------
# High-level convenience function
# ---------------------------------------------------------------------------

def capture_history_clip(
    camera: CameraConfig,
    channel: int,
    start_time: str,
    *,
    duration_sec: float = 4.0,
    subtype: int = 0,
    output_dir: Optional[Path] = None,
) -> CaptureResult:
    """
    Shortcut: capture a single playback/history clip.

    This is the simplest way to use the capture stage::

        from pipeline.config import get_camera_config, get_temp_dir
        from pipeline.capture import capture_history_clip

        cam    = get_camera_config("CAMERA_URL_1")
        result = capture_history_clip(
            camera     = cam,
            channel    = 78,
            start_time = "2026_06_02_12_00_00",
        )
        print(result)          # ✅ CaptureResult(...)
        print(result.clip_path) # C:\\...\\temp\\ch78_playback_...mp4

    Args:
        camera:       Camera config (from ``get_camera_config``).
        channel:      DVR channel number.
        start_time:   Playback start (``"YYYY_MM_DD_HH_MM_SS"``).
        duration_sec: Seconds to capture (default ``4.0``).
        subtype:      Stream subtype (default ``0``).
        output_dir:   Where to save. Defaults to project ``temp/``.
    """
    from pipeline.config import get_temp_dir   # lazy to avoid circular import

    request = CaptureRequest(
        camera=camera,
        channel=channel,
        subtype=subtype,
        mode=StreamMode.PLAYBACK,
        start_time=start_time,
        duration_sec=duration_sec,
        output_dir=output_dir or get_temp_dir(),
    )
    return capture_clip(request)


def capture_live_clip(
    camera: CameraConfig,
    channel: int,
    *,
    duration_sec: float = 4.0,
    subtype: int = 0,
    output_dir: Optional[Path] = None,
) -> CaptureResult:
    """
    Shortcut: capture a single live/realtime clip.

    Usage::

        from pipeline.config import get_camera_config
        from pipeline.capture import capture_live_clip

        cam    = get_camera_config("CAMERA_URL_1")
        result = capture_live_clip(camera=cam, channel=78)

    Args:
        camera:       Camera config.
        channel:      Channel number.
        duration_sec: Seconds to capture (default ``4.0``).
        subtype:      Stream subtype (default ``0``).
        output_dir:   Where to save. Defaults to project ``temp/``.
    """
    from pipeline.config import get_temp_dir

    request = CaptureRequest(
        camera=camera,
        channel=channel,
        subtype=subtype,
        mode=StreamMode.LIVE,
        start_time=None,
        duration_sec=duration_sec,
        output_dir=output_dir or get_temp_dir(),
    )
    return capture_clip(request)
