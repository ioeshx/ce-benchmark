from typing import Dict, List

import torch
from T2IBenchmark import calculate_clip_score

from typing import Tuple
import clip
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import numpy as np

class _CaptionImageDataset(Dataset):
    def __init__(self, image_paths: List[str], captions_mapping: Dict[str, str], preprocess) -> None:
        self.image_paths = image_paths
        self.captions_mapping = captions_mapping
        self.preprocess = preprocess

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, str]:
        path = self.image_paths[idx]
        image = Image.open(path).convert("RGB")
        return self.preprocess(image), self.captions_mapping[path]
    

def run_clip_score(
    image_paths: List[str],
    captions_mapping: Dict[str, str],
    device: str,
    batch_size: int,
    seed: int,
    workers: int,
) -> Dict[str, float]:
    # Keep batch/worker defaults conservative for better OOM safety.
    effective_batch_size = batch_size if batch_size is not None else 16
    effective_workers = workers if workers is not None else 0

    clip_kwargs = {
        "device": device,
        "verbose": True,
        "batch_size": effective_batch_size,
        "dataloader_workers": effective_workers,
    }
    if seed is not None:
        clip_kwargs["seed"] = seed

    # # Use T2IBenchmark implementation, but disable grad globally in this call to
    # # avoid autograd graph accumulation and reduce GPU memory usage.
    # with torch.inference_mode():
    #     score = calculate_clip_score(
    #         image_paths,
    #         captions_mapping=captions_mapping,
    #         **clip_kwargs,
    #     )
    # Previous local implementation is intentionally kept below for reference.
    
    
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.float()  # 强制转换为 float32，避免半精度卷积报错
    model.eval()
    dataset = _CaptionImageDataset(image_paths, captions_mapping, preprocess)
    dataloader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=effective_workers,
        pin_memory=str(device).startswith("cuda"),
    )
    score_sum = 0.0
    sample_count = 0
    with torch.inference_mode():
        for images, captions in dataloader:
            images = images.to(device, non_blocking=True)
            tokens = clip.tokenize(list(captions), truncate=True).to(device)
            image_features = model.encode_image(images)
            text_features = model.encode_text(tokens)
            image_features = image_features / image_features.norm(dim=1, keepdim=True)
            text_features = text_features / text_features.norm(dim=1, keepdim=True)
            batch_scores = (image_features * text_features).sum(dim=1)
            score_sum += batch_scores.float().sum().item()
            sample_count += batch_scores.shape[0]
    
    if sample_count == 0:
        raise ValueError("No samples available for CLIP score")
    score_mean = score_sum / sample_count
    print("Clip Score mean:{:.6f}".format(score_mean))
    return {"value": float(score_mean)}
