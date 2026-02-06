from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    images_root: str
    prompts_csv: Optional[str]
    output_json: str
    metrics: List[str]
    device: str = "cuda"
    batch_size: int
    workers: int
    seed: int
    prompt_col: str = "prompt"
    id_col: str = "case_number"
    image_col: Optional[str] = None
    fid_ref: Optional[str] = None
    lpips_original: Optional[str] = None
    lpips_edited: Optional[str] = None
    lpips_net: str = "alex"
    lpips_version: str = "0.1"
    aesthetic_model_path: Optional[str] = None
    nudity_threshold: float = 0.2
    nudity_model_path: Optional[str] = None
    nudity_resolution: int = 320
    q16_repo: Optional[str] = None
    q16_output_tag: str = "q16_eval"
    q16_python: Optional[str] = None
    extra: dict = field(default_factory=dict)
