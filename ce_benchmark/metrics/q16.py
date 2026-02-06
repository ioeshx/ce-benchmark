import csv
import os
import subprocess
import sys
from typing import Dict, Optional

from ..io import DEFAULT_EXTS


def _expected_output_csv(repo_path: str, output_tag: str) -> str:
    return os.path.join(
        repo_path,
        "data",
        output_tag,
        "inappropriate",
        "Clip_ViT-L",
        "inappropriate_images.csv",
    )


def run_q16(
    images_root: str,
    q16_repo: Optional[str],
    output_tag: str,
    python_exe: Optional[str],
) -> Dict[str, float]:
    if not q16_repo:
        raise ValueError("Q16 requires --q16-repo")

    script_path = os.path.join(
        q16_repo, "main", "clip_classifier", "classify", "inference_images.py"
    )
    if not os.path.exists(script_path):
        raise FileNotFoundError(f"Q16 script not found: {script_path}")

    python_cmd = python_exe or sys.executable
    cmd = [
        python_cmd,
        script_path,
        "--input_folder",
        images_root,
        "--output_folder",
        output_tag,
    ]
    subprocess.run(cmd, cwd=q16_repo, check=True)

    csv_path = _expected_output_csv(q16_repo, output_tag)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Q16 output not found: {csv_path}")

    flagged = set()
    with open(csv_path, "r", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            filename = row[-1]
            flagged.add(filename)

    total = 0
    for name in os.listdir(images_root):
        if name.lower().endswith(DEFAULT_EXTS):
            total += 1
    rate = len(flagged) / total if total else 0.0
    return {
        "q16_rate": float(rate),
        "q16_count": int(len(flagged)),
        "total": int(total),
        "csv": csv_path,
    }
