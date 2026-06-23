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

if os.path.exists("custom_x3d_hybrid.pth"):
    MODEL_PATH = "custom_x3d_hybrid.pth"

print(f"[SETUP] Loading weights from {MODEL_PATH}...")
state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
checkpoint_classes = state_dict["blocks.5.proj.weight"].shape[0]

if checkpoint_classes != NUM_CLASSES:
    print(f"[SETUP WARNING] Class count mismatch: checkpoint '{MODEL_PATH}' has {checkpoint_classes} classes, but {CLASSES_PATH} has {NUM_CLASSES} classes.")
    print("[SETUP WARNING] The model's classification head (projection layer) will be randomly initialized. Please run training to fine-tune the model.")
    if "blocks.5.proj.weight" in state_dict:
        del state_dict["blocks.5.proj.weight"]
    if "blocks.5.proj.bias" in state_dict:
        del state_dict["blocks.5.proj.bias"]
    MODEL.load_state_dict(state_dict, strict=False)
else:
    MODEL.load_state_dict(state_dict)

MODEL.to(DEVICE)
MODEL.eval()

print("[SETUP] Model loaded and ready")

# --- Load transform ---
TRANSFORM = get_transform()
print("[SETUP] Transform loaded")

SOFTMAX = torch.nn.Softmax(dim=1)

print("[SETUP] Setup complete")
