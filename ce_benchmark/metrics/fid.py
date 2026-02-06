from typing import Dict

from T2IBenchmark import calculate_fid
from T2IBenchmark.datasets import get_coco_fid_stats


def run_fid(
    images_root: str,
    ref: str,
    device: str,
    seed: int,
    batch_size: int,
    workers: int,
) -> Dict[str, float]:
    if ref == "coco":
        ref_input = get_coco_fid_stats()
    else:
        ref_input = ref
    fid_kwargs = {
        "device": device,
        "verbose": True,
    }
    if seed is not None:
        fid_kwargs["seed"] = seed
    if batch_size is not None:
        fid_kwargs["batch_size"] = batch_size
    if workers is not None:
        fid_kwargs["dataloader_workers"] = workers

    fid, _ = calculate_fid(images_root, ref_input, **fid_kwargs)
    return {"value": float(fid)}
