#video + image dataset
import os
import json
import torch
import torch.nn as nn
import pathlib
import random
from torch.utils.data import DataLoader, Dataset
from torchvision.io import read_image # <--- Added for loading images

# PytorchVideo imports
try:
    from pytorchvideo.data.encoded_video import EncodedVideo
except ImportError:
    raise ImportError("Pytorchvideo not installed. Please run '!pip install pytorchvideo'")

# Import from your setup file
from model_setup import get_model, get_transform, DEVICE, model_transform_params, MODEL_NAME

# --- USER CONFIG ---
TRAIN_ROOT = "dataset/train"
VAL_ROOT   = "dataset/val"

BATCH_SIZE = 4
LEARNING_RATE = 0.001
EPOCHS = 10

# --- 1. THE CUSTOM DATASET CLASS (Updated for Hybrid Input) ---
class CustomVideoDataset(Dataset):
    def __init__(self, root_path, class_to_idx, transform, clip_duration, target_frames):
        super().__init__()
        self.transform = transform
        self.clip_duration = clip_duration
        self.target_frames = target_frames # <--- Needed to make static videos "long" enough
        self.data = []

        # 1. Scan folders for BOTH Videos and Images
        for cls_name, cls_idx in class_to_idx.items():
            cls_dir = pathlib.Path(root_path) / cls_name
            if not cls_dir.exists():
                continue

            # Find video AND image files
            files = (
                list(cls_dir.glob("*.mp4")) + list(cls_dir.glob("*.avi")) + list(cls_dir.glob("*.mov")) +
                list(cls_dir.glob("*.jpg")) + list(cls_dir.glob("*.png")) + list(cls_dir.glob("*.jpeg"))
            )
            for f in files:
                self.data.append((str(f), cls_idx))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        video_path, label = self.data[idx]
        ext = os.path.splitext(video_path)[1].lower()

        try:
            # --- CASE A: IT IS AN IMAGE (Treat as Static Video) ---
            if ext in ['.jpg', '.jpeg', '.png']:
                # 1. Load Image (C, H, W) in uint8
                # image = read_image(video_path)

                # # 2. Create "Static Video" by repeating the image T times
                # # Result shape: (C, T, H, W)
                # # We repeat it 'target_frames' times so the temporal subsampler has enough data
                # video_tensor = image.unsqueeze(1).repeat(1, self.target_frames, 1, 1)

                # # 3. Create the dict structure expected by transforms
                # video_data = {"video": video_tensor}
                image = read_image(video_path)  # (C, H, W)

                # 🔥 FORCE RGB
                if image.shape[0] == 4:        # RGBA → RGB
                    image = image[:3, :, :]
                elif image.shape[0] == 1:      # Grayscale → RGB
                    image = image.repeat(3, 1, 1)

                video_tensor = image.unsqueeze(1).repeat(1, self.target_frames, 1, 1)
                video_data = {"video": video_tensor}


            # --- CASE B: IT IS A VIDEO ---
            else:
                video = EncodedVideo.from_path(video_path)
                video_duration = video.duration

                # Random Crop (Temporal) logic
                if video_duration <= self.clip_duration:
                    start_sec = 0
                else:
                    max_start = video_duration - self.clip_duration
                    start_sec = random.uniform(0, max_start)

                end_sec = start_sec + self.clip_duration

                # Get Clip
                video_data = video.get_clip(start_sec=start_sec, end_sec=end_sec)

            # --- COMMON TRANSFORM STEP ---
            # Both Image and Video tensors go through the exact same resizing/normalization
            if self.transform:
                video_data = self.transform(video_data)

            return {"video": video_data["video"], "label": label}

        except Exception as e:
            print(f"Error loading {video_path}: {e}")
            return self.__getitem__(random.randint(0, len(self) - 1))

# --- MAIN TRAINING SCRIPT ---

def start_training():
    print(f"--- Starting Hybrid Training (Image+Video) on Device: {DEVICE} ---")
    print(f"--- Model: {MODEL_NAME} ---")

    # 1. Setup Classes
    if not os.path.exists(TRAIN_ROOT):
        raise FileNotFoundError(f"Train Path not found: {TRAIN_ROOT}")

    class_names = sorted([d.name for d in pathlib.Path(TRAIN_ROOT).iterdir() if d.is_dir()])
    if len(class_names) == 0:
        raise ValueError(f"No class folders found in {TRAIN_ROOT}")

    class_to_idx = {cls_name: i for i, cls_name in enumerate(class_names)}
    id_to_classname = {i: cls_name for i, cls_name in enumerate(class_names)}

    print(f"Classes Found ({len(class_names)}): {class_names}")

    with open("classes.json", "w") as f:
        json.dump(id_to_classname, f)

    # 2. Setup Config & Transforms
    transform = get_transform()
    params = model_transform_params[MODEL_NAME]

    # Calculate duration and target frames
    target_frames = params["num_frames"]
    clip_duration = target_frames * params["sampling_rate"] / 30.0

    # 3. Initialize Custom Datasets
    print("Initializing Training Dataset...")
    train_dataset = CustomVideoDataset(
        root_path=TRAIN_ROOT,
        class_to_idx=class_to_idx,
        transform=transform,
        clip_duration=clip_duration,
        target_frames=target_frames # <--- Pass this down
    )
    print(f"Total Training Files (Img+Vid): {len(train_dataset)}")

    # Validation
    val_loader = None
    if os.path.exists(VAL_ROOT):
        print("Initializing Validation Dataset...")
        val_dataset = CustomVideoDataset(
            root_path=VAL_ROOT,
            class_to_idx=class_to_idx,
            transform=transform,
            clip_duration=clip_duration,
            target_frames=target_frames
        )
        print(f"Total Validation Files: {len(val_dataset)}")
        if len(val_dataset) > 0:
            val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    if len(train_dataset) == 0:
        raise ValueError("No files found! Check your paths.")

    # 4. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    # 5. Load Model & Optimizer
    model = get_model(num_classes=len(class_names))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

    # 6. Training Loop
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        print(f"\nEpoch {epoch+1}/{EPOCHS}")

        for i, batch in enumerate(train_loader):
            inputs = batch["video"].to(DEVICE)
            labels = batch["label"].to(DEVICE)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

            if i % 10 == 0:
                print(f"   Batch {i}/{len(train_loader)}: Loss {loss.item():.4f}")

        train_acc = 100 * correct_train / total_train
        avg_loss = running_loss / len(train_loader)
        print(f"   >> Train Accuracy: {train_acc:.2f}% | Loss: {avg_loss:.4f}")

        if val_loader:
            model.eval()
            correct_val = 0
            total_val = 0
            val_loss = 0.0

            with torch.no_grad():
                for batch in val_loader:
                    inputs = batch["video"].to(DEVICE)
                    labels = batch["label"].to(DEVICE)

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()

                    _, predicted = torch.max(outputs.data, 1)
                    total_val += labels.size(0)
                    correct_val += (predicted == labels).sum().item()

            if total_val > 0:
                val_acc = 100 * correct_val / total_val
                print(f"   >> Val Accuracy: {val_acc:.2f}% | Val Loss: {val_loss/len(val_loader):.4f}")

    # 7. Save Model
    torch.save(model.state_dict(), "custom_x3d_hybrid.pth")
    print("\nTraining Complete! Model saved as 'custom_x3d_hybrid.pth'")

if __name__ == '__main__':
    main()