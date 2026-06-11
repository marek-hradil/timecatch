import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
from config import load
from dataset import Dataset, CorruptDataset
from inference import Model
from results import Results


def main(cfg: dict) -> None:
    model = Model(cfg["model"]["path"], tensor_parallel_size=cfg["model"].get("tensor_parallel_size", 1), video_mode=cfg["model"].get("video_mode", False), max_model_len=cfg["model"].get("max_model_len"), max_image_size=cfg["model"].get("max_image_size"))
    base = Dataset(cfg["dataset"]["path"], cfg["dataset"]["seq_len_min"], cfg["dataset"]["seq_len_max"], seed=cfg["seed"], manifest=cfg["dataset"].get("manifest"))
    scenarios = CorruptDataset(base).sample_binary(cfg.get("n"))
    exp_name = f"corrupt_detect_{cfg['dataset']['name']}_{cfg['model']['name']}"
    results = Results(exp_name, cfg["model"]["name"], total=len(scenarios), out_dir=cfg.get("out_dir", "outputs"))

    include_scene_desc = cfg.get("include_scene_descriptions", True)
    batch_size = cfg.get("batch_size", len(scenarios))
    for i in range(0, len(scenarios), batch_size):
        batch = scenarios[i:i + batch_size]
        requests = [(s.frames, cfg["system_prompt"], s.scene_description if include_scene_desc else None) for s in batch]
        answers = model.ask_batch(requests, pattern=r"yes|no")
        for s, answer in zip(batch, answers):
            results.log(s.name, len(s.frames), s.anomaly, answer)

    results.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(load(args.config))
