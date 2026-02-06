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
    fid, _ = calculate_fid(
        images_root,
        ref_input,
        device=device,
        seed=seed,
        batch_size=batch_size,
        dataloader_workers=workers,
        verbose=True,
    )
    return {"value": float(fid)}
