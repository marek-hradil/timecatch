import yaml
from pathlib import Path

CONFIG_ROOT = Path(__file__).parent / "config"


def load(experiment_path: str) -> dict:
    with open(experiment_path) as f:
        exp = yaml.safe_load(f)

    with open(CONFIG_ROOT / "models" / f"{exp['model']}.yaml") as f:
        model = yaml.safe_load(f)

    with open(CONFIG_ROOT / "datasets" / f"{exp['dataset']}.yaml") as f:
        dataset = yaml.safe_load(f)

    return {
        **{k: v for k, v in exp.items() if k not in ("model", "dataset")},
        "model": model,
        "dataset": dataset,
    }
