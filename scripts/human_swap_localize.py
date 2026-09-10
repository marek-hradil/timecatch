"""
Run swap-localize inference on the exact stimuli that human annotators saw.

Reads a human_manifest JSON (produced by export_human_subset.py) and evaluates
the model on each unique stimulus, preserving the same frame order humans saw.
Results are written in the standard 4-column CSV format (scenario, seq_len,
ground_truth, answer) so existing consolidate.py and plotting scripts work.

Config fields (in addition to standard model/dataset/slurm/system_prompt):
  human_manifest: path to human_subset/swap_localize_<dataset>.json
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json

from config import load
from dataset import Dataset
from inference import Model, Prompt, build_messages
from results import Results


def _swap_localize_pattern(n: int) -> str:
    """Constrained-decoding pattern for 1-based adjacent swap pairs."""
    return "|".join(f"{i},{i+1}" for i in range(1, n))


def main(cfg: dict) -> None:
    model = Model(
        cfg["model"]["path"],
        tensor_parallel_size=cfg["model"].get("tensor_parallel_size", 1),
        video_mode=cfg["model"].get("video_mode", False),
        max_model_len=cfg["model"].get("max_model_len"),
        enforce_eager=cfg["model"].get("enforce_eager", False),
    )
    model_id = cfg["model"]["path"]
    video_mode = cfg["model"].get("video_mode", False)

    manifest_path = cfg["human_manifest"]
    with open(manifest_path) as f:
        manifest = json.load(f)

    base = Dataset(
        cfg["dataset"]["path"],
        cfg["dataset"]["seq_len_min"],
        cfg["dataset"]["seq_len_max"],
        seed=cfg["seed"],
        manifest=cfg["dataset"].get("manifest"),
    )

    scenarios = []
    for stimulus in manifest["stimuli"]:
        s = base.load_explicit_stimulus(
            stimulus["scene_name"],
            stimulus["shown_order"],
            stimulus["ground_truth"],
        )
        scenarios.append(s)

    exp_name = f"human_swap_localize_{cfg['dataset']['name']}_{cfg['model']['name']}"
    results = Results(
        exp_name,
        cfg["model"]["name"],
        total=len(scenarios),
        out_dir=cfg.get("out_dir", "outputs"),
    )

    include_scene_desc = cfg.get("include_scene_descriptions", True)
    batch_size = cfg.get("batch_size", len(scenarios))
    for i in range(0, len(scenarios), batch_size):
        batch = scenarios[i:i + batch_size]
        prompts = [
            Prompt(s.frames, cfg["system_prompt"], s.scene_description if include_scene_desc else None)
            for s in batch
        ]
        requests = [(build_messages(p, model_id, video_mode), p.images) for p in prompts]
        pattern = _swap_localize_pattern(max(len(s.frames) for s in batch))
        answers = model.ask_batch(requests, pattern=pattern)
        for s, answer in zip(batch, answers):
            results.log(s.name, len(s.frames), s.position, answer)

    results.summary()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config")
    args = parser.parse_args()
    main(load(args.config))
