import base64
import json
import torch
import io
import av
import numpy as np
import multiprocessing
import queue
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, UploadFile, File, APIRouter
from fastapi.responses import JSONResponse
from setup import MODEL, TRANSFORM, ID_TO_CLASS, SOFTMAX, DEVICE
from torchvision.transforms.functional import to_tensor
from train import start_training

app = FastAPI()

router = APIRouter()

TOTAL_DURATION = 3

@router.get("/train")
def train():
    """
        make sure dataset is up to date with train,val,test folders
        and all classes added in sequence in a classes.json file
    """
    start_training()

@router.post("/predict/video")
async def predict_from_video(file: UploadFile = File(...)):
    try:
        print("[HTTP] Received video upload")

        # 1. Read uploaded file into memory
        video_bytes = await file.read()
        file_obj = io.BytesIO(video_bytes)

        # 2. Open video container from RAM
        container = av.open(file_obj)
        total_duration = TOTAL_DURATION

        print(f"[HTTP] Video loaded. Duration: {total_duration:.2f}s")

        all_results = []

        # 3. Progressive prediction
        for sec in range(1, int(total_duration) + 1):
            clip_data = read_clip_from_memory(container, end_sec=sec)

            if clip_data is None:
                continue

            clip_data = TRANSFORM(clip_data)
            inputs = clip_data["video"].to(DEVICE)

            with torch.no_grad():
                preds = MODEL(inputs.unsqueeze(0))
                probs = SOFTMAX(preds)

            top_k = min(5, len(ID_TO_CLASS))
            scores, indices = probs.topk(top_k)

            predictions = []
            for idx, score in zip(indices[0], scores[0]):
                predictions.append({
                    "label": ID_TO_CLASS[int(idx)],
                    "confidence": round(score.item() * 100, 2)
                })

            all_results.append({
                "duration": sec,
                "predictions": predictions
            })

        container.close()
        file_obj.close()

        print("[HTTP] Inference complete")

        return JSONResponse(content={
            "filename": file.filename,
            "duration": round(total_duration, 2),
            "results": all_results
        })

    except Exception as e:
        print("[HTTP ERROR]", e)
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )



print("[API] FastAPI initialized - In-Memory Mode")

def get_video_duration(container):
    """Safe retrieval of duration from PyAV container."""
    stream = container.streams.video[0]
    if stream.duration:
        return float(stream.duration * stream.time_base)
    if container.duration:
        return float(container.duration / av.time_base)
    return 0.0

import math

def read_clip_from_memory(container, end_sec):
    """
    Decodes frames from 0 to end_sec directly from RAM.
    Corrected to handle stream start_time offset (Fixes prediction mismatch).
    """
    frames = []
    video_stream = container.streams.video[0]
    
    # 1. Get the actual start time of the stream (handle mobile video offsets)
    start_offset = video_stream.start_time if video_stream.start_time else 0
    
    # 2. Calculate end timestamp relative to that offset
    # We use stream.time_base to convert seconds to 'Presentation Time Stamps' (PTS)
    end_pts = start_offset + int(end_sec / video_stream.time_base)
    
    # 3. Seek to the absolute start
    container.seek(start_offset, stream=video_stream)
    
    for frame in container.decode(video=0):
        # 4. Filter frames strictly within the requested window
        if frame.pts < start_offset:
            continue
        if frame.pts > end_pts:
            break
        
        # Convert to Numpy RGB
        img = frame.to_rgb().to_ndarray()
        frames.append(img)

    if not frames:
        return None

    # Stack and format for PyTorch
    buffer = np.stack(frames)
    video_tensor = torch.from_numpy(buffer)
    # Permute to (Channel, Time, Height, Width)
    video_tensor = video_tensor.permute(3, 0, 1, 2)

    return {"video": video_tensor, "audio": None}

@app.websocket("/ws/predict")
async def websocket_predict(websocket: WebSocket):
    await websocket.accept()
    print("[WS] Client connected")

    try:
        while True:
            # 1. Receive Data
            raw_data = await websocket.receive_text()
            data = json.loads(raw_data)
            b64_string = data["bytes"]
            
            # 2. Decode Base64 -> Raw Bytes (In Memory)
            video_bytes = base64.b64decode(b64_string)
            
            # 3. Create In-Memory File Object
            # This acts exactly like a file but lives 100% in RAM
            file_obj = io.BytesIO(video_bytes)
            
            try:
                # 4. Open Container with PyAV (No disk I/O)
                container = av.open(file_obj)
                total_duration = get_video_duration(container)
                
                print(f"[WS] Video loaded in RAM. Duration: {total_duration:.2f}s")
                
                # 5. Progressive Prediction Loop
                # We iterate from 1 second up to the total duration
                for sec in range(1, TOTAL_DURATION):
                    
                    # Extract clip from memory
                    clip_data = read_clip_from_memory(container, end_sec=sec)
                    
                    if clip_data is None:
                        continue

                    # Transform (Resize, Crop, Normalize)
                    clip_data = TRANSFORM(clip_data)
                    inputs = clip_data["video"].to(DEVICE)

                    # Inference
                    with torch.no_grad():
                        # Add batch dimension: (C, T, H, W) -> (1, C, T, H, W)
                        preds = MODEL(inputs.unsqueeze(0))
                        probs = SOFTMAX(preds)

                    # Get Top 5 Predictions
                    top_k = min(5, len(ID_TO_CLASS))
                    scores, indices = probs.topk(top_k)

                    predictions = []
                    for idx, score in zip(indices[0], scores[0]):
                        predictions.append({
                            "label": ID_TO_CLASS[int(idx)],
                            "confidence": round(score.item() * 100, 2)
                        })

                    result = {
                        "duration": sec,
                        "predictions": predictions
                    }

                    # print(f"[WS] Sent result for {sec}s")
                    await websocket.send_json(result)
                
                # Close the container for this specific request
                container.close()
                print("[WS] Inference complete")

            except Exception as e:
                print(f"[WS ERROR] processing video: {e}")
                await websocket.send_json({"error": "Failed to process video data"})
            
            finally:
                # Close the BytesIO buffer
                file_obj.close()

    except WebSocketDisconnect:
        print("[WS] Client disconnected")

    except Exception as e:
        print("[WS CRITICAL]", e)
        await websocket.send_json({"error": str(e)})


from pydantic import BaseModel
from typing import List
import os
import time
from pathlib import Path
from datetime import datetime, timedelta

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from pipeline.config import CameraConfig, get_temp_dir
from pipeline.models import CaptureRequest, StreamMode
from pipeline.capture import capture_clip
from pipeline.detect import extract_person_clips
from pipeline.label import label_person_clips


class GenerateDatasetRequest(BaseModel):
    activity_list: List[str]
    dataset_path: str
    camera_url: str
    camera_user: str
    camera_pass: str
    channel: int = 22
    subtype: int = 0
    mode: str = "playback"
    starttime: str = "2026_06_22_12_00_00"
    clip_duration: float = 4.0
    skip_duration: float = 0.0
    total_duration: float = 200.0
    label_delay: float = 1.0   # seconds between labeling calls to avoid rate limits


@router.post("/generate_dataset")
def generate_dataset(req: GenerateDatasetRequest):
    try:
        print("[HTTP] Generating dataset in pipeline mode...")

        # Pass activity classes to the label module
        os.environ["ACTIVITY_CLASSES"] = ",".join(req.activity_list)

        cam = CameraConfig(
            camera_id="api_camera",
            base_url=req.camera_url,
            username=req.camera_user,
            password=req.camera_pass
        )

        raw_dir = get_temp_dir() / "raw_captures"
        raw_dir.mkdir(parents=True, exist_ok=True)

        def next_start_time(base: str, offset_seconds: int) -> str:
            dt = datetime.strptime(base, "%Y_%m_%d_%H_%M_%S")
            dt += timedelta(seconds=offset_seconds)
            return dt.strftime("%Y_%m_%d_%H_%M_%S")

        stream_mode = StreamMode.PLAYBACK if req.mode.lower() == "playback" else StreamMode.LIVE
        cycle_duration = req.clip_duration + req.skip_duration
        num_captures = max(1, int(req.total_duration / cycle_duration))
        print(f"[HTTP] Total duration: {req.total_duration}s → {num_captures} clips of {req.clip_duration}s each, skipping {req.skip_duration}s between.")

        current_time = req.starttime
        total_persons = 0
        cumulative_label_stats: dict = {}

        for i in range(num_captures):
            print(f"\n--- Clip {i+1}/{num_captures} ---")
            capture_req = CaptureRequest(
                camera=cam,
                channel=req.channel,
                subtype=req.subtype,
                mode=stream_mode,
                start_time=current_time,
                duration_sec=req.clip_duration,
                output_dir=raw_dir
            )
            result = capture_clip(capture_req)

            if not result.is_success:
                print(f"  Capture failed for clip {i+1}. Skipping.")
                current_time = next_start_time(current_time, int(req.clip_duration + req.skip_duration))
                continue

            print(f"  Captured → {result.clip_path}")

            # Detect persons in this single clip
            detection = extract_person_clips(
                source_clip_path=result.clip_path,
                source_clip_id=result.clip_id,
                output_dir=get_temp_dir() / "person_clips",
                model_size="n",
                conf_threshold=0.40,
                min_frames=10
            )

            person_clips = detection.persons
            print(f"  Detected {len(person_clips)} person clips.")
            total_persons += len(person_clips)

            # Label each person clip immediately, then save to dataset
            dataset_dir = Path(req.dataset_path)
            for j, person_clip_path in enumerate(person_clips):
                # label_person_clips expects a list; pass one clip at a time
                stats = label_person_clips([person_clip_path], dataset_dir)
                # Accumulate class counts
                for cls, count in stats.items():
                    cumulative_label_stats[cls] = cumulative_label_stats.get(cls, 0) + count
                print(f"    Person clip {j+1} labelled → {stats}")

                # Throttle API calls
                time.sleep(req.label_delay)

            # Advance the timestamp for the next capture
            current_time = next_start_time(current_time, int(req.clip_duration + req.skip_duration))

        print("\n[HTTP] Pipeline complete.")
        return JSONResponse(content={
            "status": "success",
            "captures_attempted": num_captures,
            "captures_succeeded": sum(1 for _ in range(num_captures) if result.is_success),  # approximate; could track exactly
            "total_persons_detected": total_persons,
            "labeling_stats": cumulative_label_stats,
            "dataset_path": str(dataset_dir)
        })

    except Exception as e:
        print("[HTTP ERROR]", e)
        return JSONResponse(status_code=500, content={"error": str(e)})


def process_excel_task(contents: bytes, filename: str):
    import pandas as pd
    import io
    
    try:
        df = pd.read_excel(io.BytesIO(contents))
        print(f"\n[PROCESS] Starting task for file: {filename}. Total rows: {len(df)}")
        
        for index, row in df.iterrows():
            print(f"\n=== [PROCESS] Processing Excel Row {index + 1}/{len(df)} ===")
            try:
                # Build the request from the row
                req = GenerateDatasetRequest(
                    activity_list=[x.strip() for x in str(row['activity_list']).split(',')],
                    dataset_path=str(row['dataset_path']),
                    camera_url=str(row['camera_url']),
                    camera_user=str(row['camera_user']),
                    camera_pass=str(row['camera_pass']),
                    channel=int(row.get('channel', 22)),
                    subtype=int(row.get('subtype', 0)),
                    mode=str(row.get('mode', 'playback')),
                    starttime=str(row['starttime']),
                    clip_duration=float(row.get('clip_duration', 4.0)),
                    skip_duration=float(row.get('skip_duration', 0.0)),
                    total_duration=float(row.get('total_duration', 200.0)),
                    label_delay=float(row.get('label_delay', 1.0))
                )
                
                # Run the dataset generation
                generate_dataset(req)
                print(f"[PROCESS] Row {index + 1} completed successfully.")
            except Exception as row_e:
                print(f"[PROCESS ROW ERROR] Row {index + 1} failed: {row_e}")
                
        print(f"\n[PROCESS] Task for {filename} completed.")
    except Exception as e:
        print(f"[PROCESS ERROR] Failed to process excel task: {e}")


# Global task queue for Excel dataset generation
excel_task_queue = queue.Queue()


def excel_queue_worker():
    ctx = multiprocessing.get_context("spawn")
    while True:
        task = excel_task_queue.get()
        if task is None:
            break
        contents, filename = task
        try:
            print(f"[QUEUE WORKER] Spawning process to run task: {filename}")
            p = ctx.Process(target=process_excel_task, args=(contents, filename))
            p.start()
            p.join()
            print(f"[QUEUE WORKER] Process for task {filename} finished.")
        except Exception as e:
            print(f"[QUEUE WORKER ERROR] Failed to run task {filename}: {e}")
        finally:
            excel_task_queue.task_done()


# Start background worker thread only in the main FastAPI process
if multiprocessing.current_process().name == "MainProcess":
    threading.Thread(target=excel_queue_worker, daemon=True).start()


@router.post("/generate_dataset_using_excel")
async def generate_dataset_using_excel(file: UploadFile = File(...)):
    try:
        contents = await file.read()
        excel_task_queue.put((contents, file.filename))
        return JSONResponse(content={
            "status": "accepted",
            "message": f"Dataset generation task for '{file.filename}' accepted and added to the processing queue."
        })
    except Exception as e:
        print("[HTTP ERROR] Failed to queue excel task:", e)
        return JSONResponse(status_code=500, content={"error": str(e)})

app.include_router(router)

