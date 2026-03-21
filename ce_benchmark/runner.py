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

# tailor according to your needs
def get_prompt_from_filename(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    # The format is prompt_number (e.g. A_photo_of_dog_001.png)
    # The prompt might contain underscores, but the last underscore separates the number
    prompt = stem.rsplit("_", 1)[0]
    prompt = prompt.replace("_", " ")
    return prompt


def _clear_cuda_cache(device: str) -> None:
    if not str(device).startswith("cuda"):
        return
    try:
        import gc
        import torch

        gc.collect()
        torch.cuda.empty_cache()
    except Exception:
        # Cache cleanup is best-effort and should never break evaluation.
        pass


def run_benchmark(config: BenchmarkConfig) -> Dict[str, object]:
    start_time = time.time()

    image_paths: List[str] = []
    if len(config.metrics) == 1 and config.metrics[0].lower() == "lpips":
        pass # LPIPS handles image collection separately
    else:
        if config.images_root is None:
            raise ValueError("images_root must be specified")
        image_paths = collect_images(config.images_root)
        if not image_paths:
            raise ValueError(f"No images found under: {config.images_root}")

    id_to_prompt = {}
    image_to_prompt = {}
    captions = {}

    if config.prompt_from_filename:
        for path in image_paths:
            captions[path] = get_prompt_from_filename(path)
    elif config.clip_category:
        for path in image_paths:
            captions[path] = config.clip_category
    elif config.prompts_csv:
        id_to_prompt, image_to_prompt = read_prompts_csv(
            config.prompts_csv,
            prompt_col=config.prompt_col,
            id_col=config.id_col,
            image_col=config.image_col,
        )
        captions = build_captions_mapping(image_paths, id_to_prompt, image_to_prompt)
    

    results: Dict[str, object] = {
        "images_root": config.images_root,
        "fid_ref": config.fid_ref,
        "lpips_original": config.lpips_original,
        "lpips_edited": config.lpips_edited,
        "prompt_from_filename": config.prompt_from_filename,
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
        _clear_cuda_cache(config.device)

    if "clip" in metrics:
        if not config.prompts_csv and not config.prompt_from_filename:
            raise ValueError("CLIP score requires prompts CSV mapping")
        if not captions:
            raise ValueError("CLIP score requires prompts CSV mapping")
        # 
        results["metrics"]["clip_score"] = run_clip_score(
            image_paths=list(captions.keys()),
            captions_mapping=captions,
            device=config.device,
            batch_size=config.batch_size,
            seed=config.seed,
            workers=config.workers,
        )
        _clear_cuda_cache(config.device)

    if "lpips" in metrics:
        if not config.lpips_original or not config.lpips_edited:
            raise ValueError("LPIPS requires --lpips-original and --lpips-edited")
        print("LPIPS, original:", config.lpips_original, "edited:", config.lpips_edited)
        pairs = match_image_pairs(config.lpips_original, config.lpips_edited)
        results["metrics"]["lpips"] = run_lpips(
            pairs=pairs,
            device=config.device,
            net=config.lpips_net,
            version=config.lpips_version,
        )
        _clear_cuda_cache(config.device)

    if "aesthetic" in metrics:
        results["metrics"]["aesthetic"] = run_aesthetic(
            image_paths=image_paths,
            device=config.device,
            model_path=config.aesthetic_model_path,
        )
        _clear_cuda_cache(config.device)

    if "nudenet" in metrics:
        results["metrics"]["nudenet"] = run_nudenet(
            image_paths=image_paths,
            threshold=config.nudity_threshold,
            batch_size=config.batch_size,
            model_path=config.nudity_model_path,
            inference_resolution=config.nudity_resolution,
        )
        _clear_cuda_cache(config.device)

    results["elapsed_sec"] = round(time.time() - start_time, 2)
    os.makedirs(os.path.dirname(config.output_json) or ".", exist_ok=True)
    with open(config.output_json, "w", encoding="utf-8") as handle:
        json.dump(results, handle, indent=2)
    return results
