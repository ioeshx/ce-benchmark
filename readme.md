# CE-benchmark: Concept Erasure  Benchmark

This benchmark evaluates generated images with a single CLI. It supports:

- FID (from [text2image-benchmark]( https://github.com/boomb0om/text2image-benchmark))
- CLIP score (from [text2image-benchmark]( https://github.com/boomb0om/text2image-benchmark))
- LPIPS (from [PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity))
- Aesthetic score (from [discus0434/aesthetic-predictor-v2-5: ](https://github.com/discus0434/aesthetic-predictor-v2-5))
- NudeNet (from [NudeNet](https://github.com/notAI-tech/NudeNet))

TODOs: adding adversarial attack

## Install

Installing requirements

```
pip install -r requirement.txt
```

this benchmark is meant to be used as a CLI, but you can also use it as a library

```
pip install .
## pip install -e . ## editable mode
```

## Usage
common args
*  `--metrics`: metrics to calculate, separated by space. Options: `fid`, `clip`, `lpips`, `aesthetic`, `nudenet`
*  `--images-root`: root directory of images (required by `fid`, `clip`, `aesthetic`, `nudenet`)
*  `--output-json`: output json file path (default: `benchmark_results.json`)
*  `--device`: device string (default: `cuda`)
*  `--batch-size`: batch size (optional; if not set, metric defaults are used)
*  `--workers`: dataloader workers (optional; if not set, metric defaults are used)
*  `--seed`: random seed (optional; if not set, metric defaults are used)

### FID

 `--fid-ref` accepts

* a directory path
* a `.npz` stats file(containing pre-computed stats)
*  `coco` (using pre-computed MSCOCO30k stats).

```
python ce-benchmark.py \
  --metrics fid \
  --images-root /path/to/img \
  --fid-ref /path/to/ref \
  --output-json results.json
```

### Clip Score

CLIP score maps prompts by `case_number` prefix or `image_path` when provided.

CLIP-specific args
*  `--prompts-csv`: CSV containing prompts and ids
*  `--prompt-col`: prompt column name (default: `prompt`)
*  `--id-col`: id column name (default: `case_number`)
*  `--image-col`: image path column name (optional; if set, use explicit paths)

```
python ce-benchmark.py \
  --metrics clip \
  --images-root /path/to/images \
  --prompts-csv /path/to/prompts.csv \
  --output-json results.json
```

### LPIPS

LPIPS pairs images by matching filenames across folders

LPIPS-specific args
*  `--lpips-original`: folder with original images
*  `--lpips-edited`: folder with edited images
*  `--lpips-net`: backbone network (`alex` by default)
*  `--lpips-version`: LPIPS version string (default: `0.1`)

```
python ce-benchmark.py \
	--metrics lpips \
	--lpips-original /path/to/original \
  	--lpips-edited /path/to/edited \
  	--output-json results.json
```

### Aesthetic score

Aesthetic-specific args
*  `--aesthetic-model-path`: custom model checkpoint path (optional)

```
python ce-benchmark.py \
	--metrics aesthetic \
	--images-root /home/shx/Code/AdaVD/results/3style/coco1k/erase/retain \
  	--output-json results.json
```

### NudeNet

NudeNet-specific args
*  `--nudity-threshold`: score threshold (default: `0.2`)
*  `--nudity-model-path`: custom model checkpoint path (optional)
*  `--nudity-resolution`: inference resolution (default: `320`)

```
python ce-benchmark.py \
	--metrics nudenet \
	--images-root /home/shx/Code/AdaVD/results/i2p_sexual/i2p_benchmark/erase/retain \
  	--output-json results.json
```

### all

you can compute multiple metrics at a single run

```
ce-benchmark \
  --images-root /path/to/images \
  --prompts-csv /path/to/prompts.csv \
  --metrics fid clip lpips aesthetic nudenet \
  --fid-ref coco \
  --lpips-original /path/to/original \
  --lpips-edited /path/to/edited \
  --output-json /path/to/results.json
```

### Prompts CSV

Only Clip Score requires prompt CSV file.The CSV should contain an id column, a prompt column and a optinal image path col (default: `id`, `prompt`, optional: `image_path`).

Args
*  `--id-col`: use this column to map prompts by case id. The id is matched to the image filename prefix before the first `_` (for `123_0.png`, id is `123`).
*  `--image-col`: use this column to map prompts by explicit image path. When set, this exact path is used to match images and takes priority over `--id-col`.
*  Example CSV columns: `id,prompt,image_path`
