import csv
import os
from typing import Dict, Iterable, List, Optional, Tuple


DEFAULT_EXTS = (".png", ".jpg", ".jpeg", ".webp")


def collect_images(root: str, exts: Tuple[str, ...] = DEFAULT_EXTS) -> List[str]:
    image_paths = []
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            if name.lower().endswith(exts):
                image_paths.append(os.path.join(dirpath, name))
    return sorted(image_paths)


def read_prompts_csv(
    csv_path: str,
    prompt_col: str,
    id_col: str,
    image_col: Optional[str] = None,
) -> Tuple[Dict[str, str], Dict[str, str]]:
    id_to_prompt: Dict[str, str] = {}
    image_to_prompt: Dict[str, str] = {}
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            prompt = row.get(prompt_col, "").strip()
            if not prompt:
                continue
            if image_col and row.get(image_col):
                image_to_prompt[row[image_col]] = prompt
            if row.get(id_col):
                id_to_prompt[row[id_col]] = prompt
    return id_to_prompt, image_to_prompt


def infer_case_id(path: str) -> str:
    base = os.path.basename(path)
    stem, _ = os.path.splitext(base)
    if "_" in stem:
        return stem.split("_")[0]
    return stem


def build_captions_mapping(
    image_paths: Iterable[str],
    id_to_prompt: Dict[str, str],
    image_to_prompt: Dict[str, str],
) -> Dict[str, str]:
    captions = {}
    for path in image_paths:
        if path in image_to_prompt:
            captions[path] = image_to_prompt[path]
            continue
        case_id = infer_case_id(path)
        if case_id in id_to_prompt:
            captions[path] = id_to_prompt[case_id]
    return captions


def match_image_pairs(
    original_root: str,
    edited_root: str,
    exts: Tuple[str, ...] = DEFAULT_EXTS,
) -> List[Tuple[str, str]]:
    originals = collect_images(original_root, exts=exts)
    edited = collect_images(edited_root, exts=exts)
    edited_map = {os.path.basename(path): path for path in edited}
    pairs = []
    for path in originals:
        name = os.path.basename(path)
        if name in edited_map:
            pairs.append((path, edited_map[name]))
    return pairs
