import base64
import json
import torch
import io
import av
import numpy as np
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



app.include_router(router)  # <-- REQUIRED


