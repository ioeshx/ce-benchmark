from typing import Dict, Iterable, List, Tuple

import lpips
import torch


def _to_device(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        return torch.device(device)
    return torch.device("cpu")


def run_lpips(
    pairs: Iterable[Tuple[str, str]],
    device: str,
    net: str,
    version: str,
) -> Dict[str, float]:
    device_obj = _to_device(device)
    loss_fn = lpips.LPIPS(net=net, version=version)
    loss_fn = loss_fn.to(device_obj)

    distances: List[float] = []
    with torch.no_grad():
        for ref_path, pred_path in pairs:
            ref = lpips.im2tensor(lpips.load_image(ref_path)).to(device_obj)
            pred = lpips.im2tensor(lpips.load_image(pred_path)).to(device_obj)
            dist = loss_fn.forward(ref, pred)
            distances.append(float(dist.item()))

    if not distances:
        raise ValueError("No LPIPS pairs matched by filename")

    return {
        "mean": float(sum(distances) / len(distances)),
        "count": int(len(distances)),
    }
