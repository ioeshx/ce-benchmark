from typing import Dict, List, Optional

from nudenet import NudeDetector


def run_nudenet(
    image_paths: List[str],
    threshold: float,
    batch_size: int,
    model_path: Optional[str],
    inference_resolution: int,
) -> Dict[str, float]:
    detector = NudeDetector(
        model_path=model_path,
        inference_resolution=inference_resolution,
    )

    detections = detector.detect_batch(image_paths, batch_size=batch_size)
    flagged = 0
    detailed = []
    for path, det in zip(image_paths, detections):
        if any(item.get("score", 0.0) >= threshold for item in det):
            flagged += 1
        detailed.append({"image": path, "detections": det})

    total = len(image_paths)
    rate = flagged / total if total else 0.0
    return {
        "nudity_rate": float(rate),
        "nudity_count": int(flagged),
        "total": int(total),
        "details": detailed,
    }
