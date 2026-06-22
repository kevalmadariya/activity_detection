"""
test.py — Full Pipeline Test: Stage 1 (Capture) → Stage 2 (Detect)
=====================================================================
Captures multiple 4-second clips directly from the RTSP camera stream
using Stage 1 (capture.py), then runs Stage 2 (detect.py) to detect
all persons and extract one cropped person-clip per tracked person.

All output clips are saved to temp/person_clips/.

Usage:
    python test.py

Config is loaded from .env:
    CAMERA_URL_1  — camera host:port
    CAMERA_USER   — RTSP username
    CAMERA_PASS   — RTSP password (URL-encoded)
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_pipeline")

# ── Paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Pipeline imports ──────────────────────────────────────────────────────────
from pipeline.config  import get_camera_config, get_temp_dir
from pipeline.models  import CaptureRequest, StreamMode
from pipeline.capture import capture_clip
from pipeline.detect  import extract_person_clips, DetectionResult
from pipeline.label   import label_person_clips

# ── Test configuration ────────────────────────────────────────────────────────
CHANNEL          = 78
SUBTYPE          = 0
CLIP_DURATION    = 4.0          # seconds per capture
NUM_CAPTURES     = 3            # number of sequential 4s clips to capture
                                # (each may contain multiple persons → 10-12 total clips)

# Playback window: capture starting from this time, stepping forward by 4s each clip
PLAYBACK_START   = "2026_06_02_12_00_00"   # change to any valid history timestamp

OUTPUT_DIR       = PROJECT_ROOT / "temp" / "person_clips"
RAW_DIR          = get_temp_dir() / "raw_captures"   # where Stage 1 saves raw clips


# ── Helpers ───────────────────────────────────────────────────────────────────

def next_start_time(base: str, offset_seconds: int) -> str:
    """
    Increment a playback start-time string by ``offset_seconds``.

    Args:
        base:           Format ``"YYYY_MM_DD_HH_MM_SS"``.
        offset_seconds: How many seconds to add.

    Returns:
        New start-time string in same format.
    """
    dt = datetime.strptime(base, "%Y_%m_%d_%H_%M_%S")
    dt += timedelta(seconds=offset_seconds)
    return dt.strftime("%Y_%m_%d_%H_%M_%S")


def print_summary(capture_count: int, all_results: list) -> None:
    """Print a rich summary of captures and person clips produced."""
    total_persons = sum(r.total_persons for r in all_results)
    successful_captures = sum(1 for r in all_results if r.is_success)

    print()
    print("=" * 72)
    print("  PIPELINE TEST RESULTS  (Stage 1 → Stage 2)")
    print("=" * 72)
    print(f"  Raw clips captured     : {capture_count}")
    print(f"  Successfully processed : {successful_captures}")
    print(f"  Total person clips     : {total_persons}")
    print("-" * 72)
    print(f"  {'SOURCE RAW CLIP':<34} {'PERSONS':>7}  STATUS")
    print("-" * 72)

    for r in all_results:
        status = "✅ OK" if r.is_success else f"❌ {(r.error or '')[:30]}"
        print(f"  {r.source_path.name:<34} {r.total_persons:>7}  {status}")

    print("-" * 72)
    print()
    print(f"  {'PERSON CLIP FILE':<52} {'FRAMES':>6}  {'SIZE(KB)':>9}")
    print("-" * 72)

    for r in all_results:
        for p in r.persons:
            size_kb = (
                round(p.clip_path.stat().st_size / 1024, 1)
                if p.clip_path.exists() else 0.0
            )
            print(f"  {p.clip_path.name:<52} {p.frame_count:>6}  {size_kb:>9.1f}")

    print("=" * 72)
    print(f"  Output → {OUTPUT_DIR}")
    print("=" * 72)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load camera config from .env ─────────────────────────────────────────
    logger.info("Loading camera config from .env …")
    cam = get_camera_config("CAMERA_URL_1")
    logger.info("Camera: %s  |  URL: %s", cam.camera_id, cam.base_url)

    # ── Stage 1: Capture NUM_CAPTURES consecutive 4s clips ───────────────────
    logger.info(
        "Starting capture — %d clips × %.0fs each  (channel=%d)",
        NUM_CAPTURES, CLIP_DURATION, CHANNEL,
    )

    capture_results = []
    start_time = PLAYBACK_START

    for i in range(1, NUM_CAPTURES + 1):
        logger.info("─── Capture %d / %d  start=%s ───", i, NUM_CAPTURES, start_time)

        request = CaptureRequest(
            camera       = cam,
            channel      = CHANNEL,
            subtype      = SUBTYPE,
            mode         = StreamMode.PLAYBACK,
            start_time   = start_time,
            duration_sec = CLIP_DURATION,
            output_dir   = RAW_DIR,
        )

        result = capture_clip(request)

        if result.is_success:
            logger.info("  ✅ Captured → %s (%.1f KB)",
                        result.clip_path.name,
                        result.clip_path.stat().st_size / 1024)
        else:
            logger.error("  ❌ Capture FAILED: %s", result.error)

        capture_results.append(result)

        # Advance start time by clip duration for next capture
        start_time = next_start_time(start_time, int(CLIP_DURATION))

    successful = [r for r in capture_results if r.is_success and r.clip_path]
    logger.info(
        "Capture complete — %d / %d clips OK",
        len(successful), NUM_CAPTURES,
    )

    if not successful:
        logger.error("No clips captured. Check camera connectivity and .env config.")
        sys.exit(1)

    # ── Stage 2: Detect persons in each captured clip ─────────────────────────
    logger.info("Starting person detection …")
    all_detection_results = []

    for i, cap_result in enumerate(successful, 1):
        logger.info(
            "─── Detecting clip %d / %d : %s ───",
            i, len(successful), cap_result.clip_path.name,
        )

        det = extract_person_clips(
            source_clip_path = cap_result.clip_path,
            source_clip_id   = cap_result.clip_id,
            output_dir       = OUTPUT_DIR,
            model_size       = "n",         # yolov8n — fastest
            conf_threshold   = 0.40,
            min_frames       = 10,
        )

        all_detection_results.append(det)

        if det.is_success:
            logger.info("  → %d person clip(s) extracted", det.total_persons)
            for p in det.persons:
                logger.info(
                    "     person_id=%-3d  frames=%-4d  file=%s",
                    p.person_id, p.frame_count, p.clip_path.name,
                )
        else:
            logger.warning("  → FAILED: %s", det.error)

    # ── Summary ───────────────────────────────────────────────────────────────
    print_summary(len(capture_results), all_detection_results)

    total_person_clips = sum(r.total_persons for r in all_detection_results)
    if total_person_clips == 0:
        logger.warning("No persons detected in any clip. Check camera angle/content.")
        sys.exit(1)

    logger.info(
        "✅ Done — %d person clips saved to %s",
        total_person_clips, OUTPUT_DIR,
    )

    # ── Stage 3: Label and organize the dataset ───────────────────────────────
    logger.info("Starting Vision LLM dataset generation …")
    all_person_clips = []
    for r in all_detection_results:
        all_person_clips.extend(r.persons)

    if not all_person_clips:
        logger.info("No person clips to label.")
        return

    dataset_output_dir = get_temp_dir() / "dataset"
    label_stats = label_person_clips(all_person_clips, dataset_output_dir)

    print()
    print("=" * 72)
    print("  STAGE 3: DATASET GENERATION RESULTS")
    print("=" * 72)
    for k, v in label_stats.items():
        print(f"  {k:<20}: {v} clips")
    print("=" * 72)
    print(f"  Dataset Directory: {dataset_output_dir}")
    print("=" * 72)


if __name__ == "__main__":
    main()
