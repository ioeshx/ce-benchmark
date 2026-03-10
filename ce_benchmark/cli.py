import argparse
import sys

from .config import BenchmarkConfig
from .runner import run_benchmark


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Unified concept erasure benchmark")
    parser.add_argument("--images-root")
    parser.add_argument("--prompts-csv")
    parser.add_argument("--output-json", default="benchmark_results.json")
    parser.add_argument(
        "--metrics",
        nargs="+",
        required=True,
        help="fid clip lpips aesthetic nudenet",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--prompt-col", default="prompt")
    parser.add_argument("--id-col", default="case_number")
    parser.add_argument("--image-col")
    ## fid config
    parser.add_argument("--fid-ref")
    # clip-score
    parser.add_argument("--prompt_from_filename", default=False, action='store_true')
    ## lpips config
    parser.add_argument("--lpips-original")
    parser.add_argument("--lpips-edited")
    parser.add_argument("--lpips-net", default="alex")
    parser.add_argument("--lpips-version", default="0.1")
    ## aesthetic score config
    parser.add_argument("--aesthetic-model-path")
    ## nudenet config
    parser.add_argument("--nudity-threshold", type=float, default=0.2)
    parser.add_argument("--nudity-model-path")
    parser.add_argument("--nudity-resolution", type=int, default=320)   
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    config = BenchmarkConfig(
        images_root=args.images_root,
        prompts_csv=args.prompts_csv,
        output_json=args.output_json,
        metrics=args.metrics,
        device=args.device,
        batch_size=args.batch_size,
        workers=args.workers,
        seed=args.seed,
        prompt_col=args.prompt_col,
        id_col=args.id_col,
        image_col=args.image_col,
        fid_ref=args.fid_ref,
        lpips_original=args.lpips_original,
        lpips_edited=args.lpips_edited,
        lpips_net=args.lpips_net,
        lpips_version=args.lpips_version,
        aesthetic_model_path=args.aesthetic_model_path,
        nudity_threshold=args.nudity_threshold,
        nudity_model_path=args.nudity_model_path,
        nudity_resolution=args.nudity_resolution,
        prompt_from_filename=args.prompt_from_filename
    )
    run_benchmark(config)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
