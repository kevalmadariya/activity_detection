# setup.py
import torch
import json
import os
from pytorchvideo.data.encoded_video import EncodedVideo

from model_setup import get_model, get_transform, DEVICE

MODEL_PATH = "custom_x3d_model.pth"
CLASSES_PATH = "classes.json"

print("[SETUP] Starting model setup...")

# --- Load class mapping ---
if not os.path.exists(CLASSES_PATH):
    raise FileNotFoundError(f"[SETUP ERROR] {CLASSES_PATH} not found")

with open(CLASSES_PATH, "r") as f:
    ID_TO_CLASS = json.load(f)
    ID_TO_CLASS = {int(k): v for k, v in ID_TO_CLASS.items()}

NUM_CLASSES = len(ID_TO_CLASS)
print(f"[SETUP] Loaded {NUM_CLASSES} classes")

# --- Load model ---
print("[SETUP] Loading model weights...")
MODEL = get_model(num_classes=NUM_CLASSES, load_pretrained=False)
MODEL.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
MODEL.to(DEVICE)
MODEL.eval()

print("[SETUP] Model loaded and ready")

# --- Load transform ---
TRANSFORM = get_transform()
print("[SETUP] Transform loaded")

SOFTMAX = torch.nn.Softmax(dim=1)

print("[SETUP] Setup complete")
