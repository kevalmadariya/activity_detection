"""
nvidia_activity_detection_test.py
===================================
A standalone test script to verify if the Nvidia Vision LLM can correctly
detect activities from a provided .mp4 video clip.

Usage:
    python nvidia_activity_detection_test.py <path_to_mp4>
"""

import sys
import base64
import json
import os
from pathlib import Path

import cv2
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv(".env")

def extract_frames_base64(video_path: Path, num_frames: int = 4) -> list[str]:
    """Extract evenly spaced frames from video and encode them as base64 JPEGs."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        print(f"Error: Cannot open video file {video_path}")
        sys.exit(1)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        print("Error: Video contains no frames.")
        sys.exit(1)

    indices = [int(i * total_frames / num_frames) for i in range(num_frames)]
    indices[-1] = min(indices[-1], total_frames - 1)

    b64_frames = []
    for idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # Resize frame if it's too large to save tokens/bandwidth
            frame = cv2.resize(frame, (512, 512))
            _, buffer = cv2.imencode('.jpg', frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            b64_frames.append(base64.b64encode(buffer).decode('utf-8'))

    cap.release()
    return b64_frames

def test_nvidia_vlm(video_path: str):
    api_key = os.getenv("NVIDIA_API_KEY")
    if not api_key or api_key == "your_nvidia_api_key_here":
        print("Error: Please set a valid NVIDIA_API_KEY in the .env file.")
        sys.exit(1)

    # Use the model from .env or default to paligemma
    model = os.getenv("VISION_LLM_MODEL", "google/paligemma-3b-mix-448")
    
    classes_str = os.getenv("ACTIVITY_CLASSES", "sitting,standing,walking")
    classes = [c.strip() for c in classes_str.split(",") if c.strip()]

    print(f"Loading video: {video_path}")
    b64_frames = extract_frames_base64(Path(video_path), num_frames=4)
    print(f"Extracted {len(b64_frames)} frames successfully.")

    invoke_url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json"
    }

    prompt = f"""Look at these frames from a video of a person.
What activity is the person doing?

You MUST respond with EXACTLY ONE of these labels:
{json.dumps(classes)}

"""

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

    print(f"\nSending request to Nvidia API using model: {model}...")
    try:
        response = requests.post(invoke_url, headers=headers, json=payload, timeout=60)
        
        if response.status_code != 200:
            print(f"API Error ({response.status_code}): {response.text}")
            sys.exit(1)
            
        data = response.json()
        raw_reply = data["choices"][0]["message"].get("content") or ""
        
        print("\n" + "="*50)
        print("  RAW MODEL RESPONSE")
        print("="*50)
        print(raw_reply)
        print("="*50)
        
        # Determine the parsed label
        reply_lower = raw_reply.strip().lower()
        detected_label = "unknown"
        for c in classes:
            if c.lower() in reply_lower:
                detected_label = c
                break
                
        print(f"\nFinal Parsed Label: => {detected_label.upper()} <=")

    except Exception as e:
        print(f"\nRequest failed: {e}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python nvidia_activity_detection_test.py <path_to_video.mp4>")
        sys.exit(1)
        
    test_nvidia_vlm(sys.argv[1])
