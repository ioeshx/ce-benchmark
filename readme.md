CLIP score maps prompts by case_number prefix or image_path when provided.

Install editable mode during development:

```
pip install -e .
```

# Concept Erasure Unified Benchmark

This benchmark evaluates pre-generated images with a single CLI. It supports:

- FID (via text2image-benchmark)
- CLIP score (via text2image-benchmark)
- LPIPS (via PerceptualSimilarity)
- Aesthetic score (aesthetic-predictor-v2-5)
- NudeNet (nudenet)
- Q16 (ml-research/Q16 repo)

## Install

```
pip install .
```

Note: Q16 is not installed via pip. Clone its repo and pass `--q16-repo`.

If you still want the legacy install path:

```
pip install -r requirement.txt
```

## Usage

### FID
```
ce-benchmark \
  --metrics fid \
  --images-root "/path/to/images" \
  --fid-ref "/path/to/ref" \
  --output-json results.json
```
### Clip Score
```
ce-benchmark \
  --metrics clip \
  --images-root /path/to/images \
  --prompts-csv /path/to/prompts.csv \
  --output-json /path/to/results.json
```


```
ce-benchmark \
  --images-root /path/to/images \
  --prompts-csv /path/to/prompts.csv \
  --metrics fid clip lpips aesthetic nudenet q16 \
  --fid-ref coco \
  --lpips-original /path/to/original \
  --lpips-edited /path/to/edited \
  --q16-repo /path/to/Q16 \
  --output-json /path/to/results.json
```

### Prompts CSV

The CSV should contain a prompt column and an id column (default: `prompt`, `case_number`).
Image filenames are matched by `case_number` as a prefix (e.g. `123_0.png`).

If you have explicit paths, add `image_path` and pass `--image-col image_path`.

## Notes

- FID `--fid-ref` accepts a directory, a `.npz` stats file, or `coco`.
- LPIPS pairs images by matching filenames across folders.
- Q16 outputs are read from `Q16/data/<output_tag>/inappropriate/Clip_ViT-L/inappropriate_images.csv`.
