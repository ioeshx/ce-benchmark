from typing import Dict, List

from T2IBenchmark import calculate_clip_score


def run_clip_score(
    image_paths: List[str],
    captions_mapping: Dict[str, str],
    device: str,
    batch_size: int,
    seed: int,
    workers: int,
) -> Dict[str, float]:
    clip_kwargs = {
        "device": device,
        "verbose": True,
    }
    if seed is not None:
        clip_kwargs["seed"] = seed
    if batch_size is not None:
        clip_kwargs["batch_size"] = batch_size
    if workers is not None:
        clip_kwargs["dataloader_workers"] = workers

    score = calculate_clip_score(
        image_paths,
        captions_mapping=captions_mapping,
        **clip_kwargs,
    )
    return {"value": float(score)}
