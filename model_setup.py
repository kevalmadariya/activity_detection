import torch
import torch.nn as nn
from torchvision.transforms import Compose, Lambda, CenterCrop, Resize

# Import the model directly from the installed library to avoid GitHub timeouts
try:
    from pytorchvideo.models.hub import x3d_l
except ImportError:
    raise ImportError("Pytorchvideo is not installed correctly. Please run '!pip install pytorchvideo'")

# --- CONFIGURATION ---
MODEL_NAME = 'x3d_l'
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Standard ImageNet/Kinetics normalization statistics
MEAN = [0.45, 0.45, 0.45]
STD = [0.225, 0.225, 0.225]

model_transform_params  = {
    "x3d_m": {"side_size": 256, "crop_size": 256, "num_frames": 16, "sampling_rate": 5},
    "x3d_l": {"side_size": 312, "crop_size": 312, "num_frames": 16, "sampling_rate": 5}
}

# --- CUSTOM UTILITIES (Version Safe) ---

class ApplyTransformToKey:
    def __init__(self, key, transform):
        self.key = key
        self.transform = transform

    def __call__(self, x):
        if self.key in x:
            x[self.key] = self.transform(x[self.key])
        return x

class UniformTemporalSubsample:
    def __init__(self, num_samples):
        self.num_samples = num_samples

    def __call__(self, x):
        t = x.shape[1]
        indices = torch.linspace(0, t - 1, self.num_samples)
        indices = torch.clamp(indices, 0, t - 1).long()
        return x[:, indices, :, :]

def normalize_video(x, mean, std):
    mean = torch.tensor(mean, device=x.device).view(-1, 1, 1, 1)
    std = torch.tensor(std, device=x.device).view(-1, 1, 1, 1)
    return (x - mean) / std

# --- MAIN FUNCTIONS ---

def get_transform():
    params = model_transform_params[MODEL_NAME]
    return ApplyTransformToKey(
        key="video",
        transform=Compose([
            UniformTemporalSubsample(params["num_frames"]),
            Lambda(lambda x: x / 255.0),
            Lambda(lambda x: normalize_video(x, MEAN, STD)),
            Resize(size=params["side_size"], antialias=True),
            CenterCrop(size=(params["crop_size"], params["crop_size"]))
        ]),
    )

def get_model(num_classes, load_pretrained=True):
    """
    Loads X3D directly from the local library.
    """
    # Direct load - Bypasses GitHub API checks
    model = x3d_l(pretrained=load_pretrained)

    # Surgery: Replace the last layer (Head)
    # in x3d_l, the last block is 'blocks[5]', and projection is 'proj'
    in_features = model.blocks[-1].proj.in_features
    model.blocks[-1].proj = nn.Linear(in_features, num_classes)

    return model.to(DEVICE)