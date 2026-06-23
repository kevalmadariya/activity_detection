"""
pipeline.label — Stage 3: Vision LLM Auto-Labeling.

Responsibilities:
    - Receives a ``PersonClip`` (or a list of them) from Stage 2.
    - Extracts representative frames.
    - Queries the Nvidia API (Vision LLM) to classify the activity.
    - Moves/copies the clip to ``temp/dataset/<activity_label>/``.
    - If no activity is confidently matched, it skips the clip.

SOLID design:
    S — Only handles labeling and dataset organization.
    I — Exposes a simple `label_person_clips` function.
"""

import base64
import json
import logging
import shutil
from pathlib import Path
from typing import List

import cv2
import requests

from pipeline.config import get_vision_llm_config
from pipeline.detect import PersonClip

logger = logging.getLogger(__name__)

def _extract_frames_base64(video_path: Path, num_frames: int = 4) -> List[str]:
    """Extracts evenly spaced frames from a video and returns them as base64 JPEG strings."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        logger.error(f"Cannot open video for labeling: {video_path}")
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        return []

    # Pick evenly spaced frame indices
    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    indices[-1] = min(indices[-1], total_frames - 1)  # Ensure last index is valid

    b64_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            b64 = base64.b64encode(buffer).decode('utf-8')
            b64_frames.append(b64)

    cap.release()
    return b64_frames

def ask_nvidia_vlm(b64_frames: List[str], api_key: str, model: str, classes: List[str]) -> str:
    """Queries Nvidia API to classify the activity in the frames with exponential backoff retries for 429."""
    import time
    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }
    
    prompt = f"""Look at these frames from a video of a person.
What activity is the person doing?

You MUST respond with EXACTLY ONE of these labels:
{json.dumps(classes)}

Note: The class working_with_machine is for worker with blue t-shirt working with any machine, hear machine is factory machine not phone or computer.
and if you are not confident enough about using phone class dont give using phone because employee might get punished based on that.

If the activity doesn't match any label or is unclear, respond with "unknown".
Respond with ONLY the label, nothing else.
"""

    # Construct content array with text and images
    content = [{"type": "text", "text": prompt}]
    for b64 in b64_frames:
        content.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
        })

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100,
        "temperature": 0.2,
        "top_p": 0.7
    }

    max_retries = 8
    backoff_factor = 1.5
    initial_delay = 5.0

    for attempt in range(max_retries):
        try:
            response = requests.post(invoke_url, headers=headers, json=payload, timeout=30)
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = float(retry_after)
                else:
                    delay = initial_delay * (backoff_factor ** attempt)
                logger.warning(f"Nvidia API returned 429 (Too Many Requests). Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
                
            response.raise_for_status()
            data = response.json()
            
            reply = data["choices"][0]["message"].get("content") or ""
            reply = reply.strip().lower()
            
            # Cleanup response (sometimes LLMs add punctuation or quotes)
            for c in classes:
                if c in reply:
                    return c
            return "unknown"
            
        except requests.exceptions.HTTPError as he:
            if response.status_code == 429:
                retry_after = response.headers.get("Retry-After")
                if retry_after and retry_after.isdigit():
                    delay = float(retry_after)
                else:
                    delay = initial_delay * (backoff_factor ** attempt)
                logger.warning(f"Nvidia API HTTPError 429. Retrying in {delay:.2f}s... (Attempt {attempt + 1}/{max_retries})")
                time.sleep(delay)
                continue
            logger.error(f"Nvidia API HTTPError: {he}")
            break
        except Exception as e:
            logger.error(f"Nvidia API request failed: {e}")
            break
            
    return "unknown"

def label_person_clips(person_clips: List[PersonClip], dataset_dir: Path) -> dict:
    """
    Takes person clips, asks the Vision LLM for the label, and copies the clip into
    dataset_dir/<label>/. Only processes up to 12 clips as requested.
    """
    try:
        api_key, model, classes = get_vision_llm_config()
    except EnvironmentError as e:
        logger.error(str(e))
        return {}

    dataset_dir.mkdir(parents=True, exist_ok=True)
    for c in classes:
        (dataset_dir / c).mkdir(exist_ok=True)
        
    stats = {c: 0 for c in classes}
    stats["unknown"] = 0
    stats["failed"] = 0

    # Process all clips provided by the Stage 2 detection
    clips_to_process = person_clips
    
    for i, clip in enumerate(clips_to_process, 1):
        logger.info(f"Labeling clip {i}/{len(clips_to_process)}: {clip.clip_path.name}")
        
        b64_frames = _extract_frames_base64(clip.clip_path, num_frames=4)
        if not b64_frames:
            logger.warning(f"  -> Failed to extract frames.")
            stats["failed"] += 1
            continue
            
        label = ask_nvidia_vlm(b64_frames, api_key, model, classes)
        logger.info(f"  -> Vision LLM Label: {label}")
        
        if label in classes:
            stats[label] += 1
            dest_dir = dataset_dir / label
            dest_path = dest_dir / clip.clip_path.name
            shutil.copy2(clip.clip_path, dest_path)
            logger.info(f"  -> Saved to {dest_path}")
        else:
            stats["unknown"] += 1
            logger.info(f"  -> Ignored (label '{label}' not in {classes})")

    return stats
