"""Config and path helpers shared by every notebook."""
import os
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(name: str) -> dict:
    """Load a file from /configs, e.g. load_yaml('baseline.yaml')."""
    with open(CONFIG_DIR / name, "r") as f:
        return yaml.safe_load(f)


def get_paths() -> dict:
    """Resolve all data/output paths. Env vars override paths.yaml."""
    cfg = load_yaml("paths.yaml")
    drive_root = Path(os.environ.get("XRETINA_DRIVE_ROOT", cfg["drive_root"]))
    raw = Path(os.environ.get("XRETINA_RAW_DIR", cfg["local_raw_dir"]))
    return {
        "drive_root": drive_root,
        "raw": raw,
        "processed": drive_root / cfg["processed_subdir"],
        "splits": drive_root / cfg["splits_subdir"],
        "checkpoints": drive_root / cfg["checkpoints_subdir"],
        "outputs": drive_root / cfg["outputs_subdir"],
    }
