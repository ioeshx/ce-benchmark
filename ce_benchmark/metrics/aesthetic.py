from typing import Dict, List, Optional

from PIL import Image
import torch

from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip


def _select_device(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def run_aesthetic(
    image_paths: List[str],
    device: str,
    model_path: Optional[str],
) -> Dict[str, float]:
    device_obj = _select_device(device)
    model, preprocessor = convert_v2_5_from_siglip(
        predictor_name_or_path=model_path,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    if device_obj.type == "cuda":
        model = model.to(torch.bfloat16).to(device_obj)
        dtype = torch.bfloat16
    else:
        model = model.to(device_obj)
        dtype = torch.float32

    scores = []
    with torch.inference_mode():
        for image_path in image_paths:
            image = Image.open(image_path).convert("RGB")
            pixel_values = preprocessor(images=image, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(device_obj, dtype=dtype)
            score = model(pixel_values).logits.squeeze().float().cpu().numpy()
            scores.append(float(score))

    return {
        "mean": float(sum(scores) / len(scores)),
        "std dev": float(torch.std(torch.tensor(scores)).item()),
        "count": float(len(scores)),
    }
