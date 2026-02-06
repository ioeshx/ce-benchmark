from typing import Dict, List, Optional

from PIL import Image
import torch

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

    return {
        "mean": float(sum(scores) / len(scores)),
        "std dev": float(torch.std(torch.tensor(scores)).item()),
        "count": float(len(scores)),
    }
