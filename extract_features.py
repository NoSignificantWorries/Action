import os
import glob

import torch
from PIL import Image


def extract_features(video_path, device, model, transform, stride=2):
    frame_paths = sorted(glob.glob(os.path.join(video_path, "*.jpg")))
    features = []
    for frame_path in frame_paths[:stride:]:
        try:
            image = Image.open(frame_path).convert('RGB')
            image = transform(image).unsqueeze(0).to(device)
            with torch.no_grad():
                feature = model(image) # [1, 512, 1, 1]
                feature = feature.squeeze() # [512]
            features.append(feature.cpu().numpy())
        except Exception as e:
            print(f"Error during frame processing {frame_path}: {e}")
            continue
    return features
