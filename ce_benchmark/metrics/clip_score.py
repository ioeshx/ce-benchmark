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
    score = calculate_clip_score(
        image_paths,
        captions_mapping=captions_mapping,
        device=device,
        seed=seed,
        batch_size=batch_size,
        dataloader_workers=workers,
        verbose=True,
    )
    return {"value": float(score)}
