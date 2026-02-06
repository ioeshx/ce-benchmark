import json
import os
import time
from typing import Dict, List

from .config import BenchmarkConfig
from .io import build_captions_mapping, collect_images, match_image_pairs, read_prompts_csv
from .metrics.aesthetic import run_aesthetic
from .metrics.clip_score import run_clip_score
from .metrics.fid import run_fid
from .metrics.lpips import run_lpips
from .metrics.nudenet import run_nudenet
from .metrics.q16 import run_q16


def run_benchmark(config: BenchmarkConfig) -> Dict[str, object]:
    start_time = time.time()
    image_paths = collect_images(config.images_root)
    if not image_paths:
        raise ValueError(f"No images found under: {config.images_root}")

    id_to_prompt = {}
    image_to_prompt = {}
    captions = {}

    if config.prompts_csv:
        id_to_prompt, image_to_prompt = read_prompts_csv(
            config.prompts_csv,
            prompt_col=config.prompt_col,
            id_col=config.id_col,
            image_col=config.image_col,
        )
        captions = build_captions_mapping(image_paths, id_to_prompt, image_to_prompt)

    results: Dict[str, object] = {
        "images_root": config.images_root,
        "prompts_csv": config.prompts_csv,
        "num_images": len(image_paths),
        "metrics": {},
    }

    metrics = [m.lower() for m in config.metrics]

    if "fid" in metrics:
        if not config.fid_ref:
            raise ValueError("FID requires --fid-ref (path, .npz, or 'coco')")
        results["metrics"]["fid"] = run_fid(
            images_root=config.images_root,
            ref=config.fid_ref,
            device=config.device,
            seed=config.seed,
            batch_size=config.batch_size,
            workers=config.workers,
        )

    if "clip" in metrics:
        if not config.prompts_csv:
            raise ValueError("CLIP score requires prompts CSV mapping")
        if not captions:
            raise ValueError("CLIP score requires prompts CSV mapping")
        results["metrics"]["clip_score"] = run_clip_score(
            image_paths=list(captions.keys()),
            captions_mapping=captions,
            device=config.device,
            batch_size=config.batch_size,
            seed=config.seed,
        )

    if "lpips" in metrics:
        if not config.lpips_original or not config.lpips_edited:
            raise ValueError("LPIPS requires --lpips-original and --lpips-edited")
        pairs = match_image_pairs(config.lpips_original, config.lpips_edited)
        results["metrics"]["lpips"] = run_lpips(
            pairs=pairs,
            device=config.device,
            net=config.lpips_net,
            version=config.lpips_version,
        )

    if "aesthetic" in metrics:
        results["metrics"]["aesthetic"] = run_aesthetic(
            image_paths=image_paths,
            device=config.device,
            model_path=config.aesthetic_model_path,
        )

    if "nudenet" in metrics:
        results["metrics"]["nudenet"] = run_nudenet(
            image_paths=image_paths,
            threshold=config.nudity_threshold,
            batch_size=config.batch_size,
            model_path=config.nudity_model_path,
            inference_resolution=config.nudity_resolution,
        )

    if "q16" in metrics:
        results["metrics"]["q16"] = run_q16(
            images_root=config.images_root,
            q16_repo=config.q16_repo,
            output_tag=config.q16_output_tag,
            python_exe=config.q16_python,
        )

    results["elapsed_sec"] = round(time.time() - start_time, 2)
    os.makedirs(os.path.dirname(config.output_json) or ".", exist_ok=True)
    with open(config.output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results
