"""
Central configuration: dataset paths, valid tasks, and env-var settings.
"""
import os
from pathlib import Path

# Resolve datasets root relative to the repo root (two levels above this file)
_REPO_ROOT = Path(__file__).parent.parent.parent
DATASETS_ROOT = Path(os.environ.get("DATASETS_ROOT", str(_REPO_ROOT / "datasets")))

DATASET_CONFIGS = {
    "CRAFT": {
        "path": DATASETS_ROOT / "CRAFT",
        "manifest": "scenes_filtered.json",
        "seq_len_min": 4,
        "seq_len_max": 8,
    },
    "CLEVRER": {
        "path": DATASETS_ROOT / "CLEVRER",
        "manifest": "scenes_filtered.json",
        "seq_len_min": 4,
        "seq_len_max": 8,
    },
    "MTL-AQA": {
        "path": DATASETS_ROOT / "MTL-AQA",
        "manifest": "scenes_filtered.json",
        "seq_len_min": 4,
        "seq_len_max": 8,
    },
    "drive_lm": {
        "path": DATASETS_ROOT / "drive_lm",
        "manifest": "scenes_filtered.json",
        "seq_len_min": 4,
        "seq_len_max": 8,
    },
}

VALID_DATASETS = set(DATASET_CONFIGS.keys())
VALID_TASKS = {"detect", "localize"}

SAMPLES_PER_PARTICIPANT: int = int(os.environ.get("SAMPLES_PER_PARTICIPANT", "15"))
N_ATTENTION_CHECKS: int = int(os.environ.get("N_ATTENTION_CHECKS", "3"))

# When set, frame URLs point directly at GCS instead of going through /frames
GCS_PUBLIC_BASE = os.environ.get("GCS_PUBLIC_BASE", "")

PROLIFIC_COMPLETION_BASE = os.environ.get(
    "PROLIFIC_COMPLETION_BASE",
    "https://app.prolific.com/submissions/complete?cc=",
)
# Completion codes keyed by "{dataset}_{task}", e.g. "CRAFT_detect"
# Read from env as PROLIFIC_CODE_CRAFT_DETECT etc.
def prolific_completion_url(dataset: str, task: str) -> str:
    env_key = f"PROLIFIC_CODE_{dataset.upper().replace('-', '_')}_{task.upper()}"
    code = os.environ.get(env_key, "")
    return PROLIFIC_COMPLETION_BASE + code
