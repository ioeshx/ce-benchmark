from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class BenchmarkConfig:
    images_root: str
    prompts_csv: Optional[str]
    output_json: str
    metrics: List[str]
    device: str = "cuda"
    batch_size: int = None
    workers: int = None
    seed: int = None
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
    extra: dict = field(default_factory=dict)
    prompt_from_filename: bool = False
    clip_category: Optional[str] = None
