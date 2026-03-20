from typing import Dict, List, Optional

from PIL import Image
import torch
import numpy as np

from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip



def run_aesthetic(
    image_paths: List[str],
    device: str,
    model_path: Optional[str],
) -> Dict[str, float]:
    
    model, preprocessor = convert_v2_5_from_siglip(
        predictor_name_or_path=model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if device.startswith("cuda"):
        model = model.to(torch.bfloat16).to(device)
        dtype = torch.bfloat16
    else:
        model = model.to(device)
        dtype = torch.float32

    scores = []
    with torch.inference_mode():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            pixel_values = preprocessor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device, dtype=dtype)
            score = model(pixel_values).logits.squeeze().float().cpu().numpy()
            scores.append(float(score))
    
    score_mean = sum(scores) / len(scores) if scores else 0.0
    score_std = np.std(scores) if scores else 0.0
    count = len(scores)
    print("Aesthetic Score mean:{:.6f}, std dev:{:.6f}, count:{}".format(score_mean, score_std, count))
    
    return {
        "mean": float(score_mean),
        "std dev": float(score_std),
        "count": float(count),
    }
