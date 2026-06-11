"""Attention rollout analysis for swap_detect on 5 MTL-AQA scenarios.

Loads Qwen3-VL-8B via transformers (not vllm), runs a forward pass with
output_attentions=True, computes attention rollout from the decision token
back to all visual tokens, and visualises which frames the model attended to.

Usage (on cluster GPU node):
  python analyze_attention.py <config_path> <scenario1> [<scenario2> ...]

Example:
  python analyze_attention.py config/experiments/swap-detect-mtl-aqa-qwen3-vl-8b.yaml \
      14_68 04_66 06_54 02_10 03_51
"""
import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
from PIL import Image as PILImage
from transformers import AutoModelForVision2Seq, AutoProcessor

from config import load
from dataset import Dataset, SwapDataset, CorruptDataset


# ── helpers ───────────────────────────────────────────────────────────────────

def attention_rollout(attentions: tuple[torch.Tensor, ...]) -> torch.Tensor:
    """Propagate attention through all layers (Abnar & Zuidema, 2020).

    Args:
        attentions: tuple of (batch=1, heads, seq_len, seq_len) per layer

    Returns:
        (seq_len, seq_len) effective attention matrix
    """
    device = attentions[0].device
    seq_len = attentions[0].shape[-1]
    result = torch.eye(seq_len, device=device)
    for attn in attentions:
        # average over heads, add identity (residual connection)
        a = attn[0].mean(dim=0)          # (seq_len, seq_len)
        a = a + torch.eye(seq_len, device=device)
        a = a / a.sum(dim=-1, keepdim=True)
        result = a @ result
    return result                        # (seq_len, seq_len)


def find_visual_token_positions(input_ids: torch.Tensor,
                                image_token_id: int) -> list[tuple[int, int]]:
    """Return (start, end) index ranges for each image's visual tokens."""
    ids = input_ids[0].tolist()
    positions = []
    i, start = 0, None
    while i < len(ids):
        if ids[i] == image_token_id:
            if start is None:
                start = i
        else:
            if start is not None:
                positions.append((start, i))
                start = None
        i += 1
    if start is not None:
        positions.append((start, len(ids)))
    return positions


def frame_attention_weights(rollout: torch.Tensor,
                            decision_pos: int,
                            frame_ranges: list[tuple[int, int]]) -> list[float]:
    """Sum attention from the decision token to each frame's token block."""
    row = rollout[decision_pos]           # (seq_len,)
    weights = []
    for start, end in frame_ranges:
        weights.append(row[start:end].sum().item())
    return weights


def spatial_heatmap(rollout: torch.Tensor,
                    decision_pos: int,
                    start: int,
                    end: int,
                    grid_hw: tuple[int, int]) -> np.ndarray:
    """Reshape the attention for one frame's tokens into a (h, w) heatmap."""
    row = rollout[decision_pos, start:end].cpu().float().numpy()
    h, w = grid_hw
    if len(row) != h * w:
        # merge_size=2 → tokens are (h//2, w//2); reshape accordingly
        h2, w2 = h // 2, w // 2
        row = row[:h2 * w2]
        arr = row.reshape(h2, w2)
    else:
        arr = row.reshape(h, w)
    return arr


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", help="experiment YAML (for dataset path / prompt)")
    parser.add_argument("scenarios", nargs="+", help="scenario names, e.g. scene_14_68")
    parser.add_argument("--task", choices=["swap", "corrupt"], default="swap",
                        help="swap_detect (default) or corrupt_detect")
    args = parser.parse_args()

    cfg = load(args.config)
    model_path = cfg["model"]["path"]
    hf_cache   = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    system_prompt = cfg["system_prompt"]

    # ── resolve model path from HF cache ──────────────────────────────────────
    hub_dir = Path(hf_cache) / "hub"
    model_dir_name = "models--" + model_path.replace("/", "--")
    snap_root = hub_dir / model_dir_name / "snapshots"
    snap = sorted(snap_root.iterdir())[-1]          # latest snapshot
    print(f"loading model from {snap}")

    # ── load dataset & select scenarios ───────────────────────────────────────
    base = Dataset(
        cfg["dataset"]["path"],
        cfg["dataset"]["seq_len_min"],
        cfg["dataset"]["seq_len_max"],
        seed=cfg["seed"],
        manifest=cfg["dataset"].get("manifest"),
    )
    ds = CorruptDataset(base) if args.task == "corrupt" else SwapDataset(base)
    all_scens = ds.sample_position(n=None)
    selected  = {s.name: s for s in all_scens if s.name in set(args.scenarios)}

    # ── load model ────────────────────────────────────────────────────────────
    print("loading processor …")
    processor = AutoProcessor.from_pretrained(str(snap), trust_remote_code=True)

    print("loading model (this takes ~30s) …")
    model = AutoModelForVision2Seq.from_pretrained(
        str(snap),
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        # Flash attention does not support output_attentions=True.
        # Eager forces standard SDPA which does.
        attn_implementation="eager",
    )
    model.eval()
    image_token_id = model.config.image_token_id

    # ── process each scenario ─────────────────────────────────────────────────
    n = len(args.scenarios)
    fig = plt.figure(figsize=(4 * 6 + 2, 4 * n))
    outer = gridspec.GridSpec(n, 1, figure=fig, hspace=0.4)

    for row_idx, name in enumerate(args.scenarios):
        if name not in selected:
            print(f"  {name}: not found, skipping")
            continue
        s = selected[name]
        frames = s.frames                            # list of PIL images
        swap_pos = s.position                        # (i, j) that were swapped

        # Build messages (same format as inference.py)
        content = [{"type": "image", "image": img} for img in frames]
        content.append({"type": "text", "text": system_prompt.strip()})
        messages = [
            {"role": "system",  "content": "You are a helpful assistant."},
            {"role": "user",    "content": content},
        ]
        text = processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = processor(
            text=[text],
            images=frames,
            return_tensors="pt",
        ).to(model.device)

        seq_len = inputs["input_ids"].shape[1]
        print(f"  {name}: seq_len={seq_len} tokens, swap at {swap_pos}")

        # Forward pass — greedy decode 1 token (the yes/no)
        with torch.no_grad():
            out = model(
                **inputs,
                output_attentions=True,
            )

        # Decision token = last position of the input (model is about to generate here)
        decision_pos = seq_len - 1
        answer_token_id = out.logits[0, decision_pos].argmax().item()
        answer = processor.tokenizer.decode([answer_token_id]).strip().lower()

        # Attention rollout
        rollout = attention_rollout(out.attentions)       # (seq_len, seq_len)

        # Locate visual token ranges for each frame
        frame_ranges = find_visual_token_positions(inputs["input_ids"], image_token_id)
        if len(frame_ranges) != len(frames):
            print(f"  WARNING: found {len(frame_ranges)} frame ranges, expected {len(frames)}")

        # Per-frame aggregate attention weight
        agg_weights = frame_attention_weights(rollout, decision_pos, frame_ranges)
        agg_arr = np.array(agg_weights)
        agg_arr = agg_arr / (agg_arr.sum() + 1e-8)

        # Spatial heatmaps (use image_grid_thw if available)
        grid_thw = inputs.get("image_grid_thw", None)

        # ── draw row ──────────────────────────────────────────────────────────
        inner = gridspec.GridSpecFromSubplotSpec(
            2, len(frames) + 1, subplot_spec=outer[row_idx],
            wspace=0.05, hspace=0.05,
        )

        # Top row: original frames with attention overlay
        for fi, (img, (start, end)) in enumerate(zip(frames, frame_ranges)):
            ax = fig.add_subplot(inner[0, fi])
            ax.imshow(img)
            if grid_thw is not None and fi < len(grid_thw):
                _, gh, gw = grid_thw[fi].tolist()
                hmap = spatial_heatmap(rollout, decision_pos, start, end, (gh, gw))
                hmap_up = PILImage.fromarray(
                    (hmap / (hmap.max() + 1e-8) * 255).astype(np.uint8)
                ).resize(img.size, PILImage.BILINEAR)
                ax.imshow(np.array(hmap_up), alpha=0.5, cmap="hot")
            border_color = "red" if fi in swap_pos else "white"
            for spine in ax.spines.values():
                spine.set_edgecolor(border_color)
                spine.set_linewidth(3)
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"F{fi+1}", fontsize=8, pad=2)

        # Bottom row: per-frame attention bar
        ax_bar = fig.add_subplot(inner[1, :len(frames)])
        colors = ["salmon" if i in swap_pos else "steelblue" for i in range(len(frames))]
        ax_bar.bar(range(len(frames)), agg_arr, color=colors)
        ax_bar.set_xticks(range(len(frames)))
        ax_bar.set_xticklabels([f"F{i+1}" for i in range(len(frames))], fontsize=7)
        ax_bar.set_ylabel("attn\nweight", fontsize=7)
        ax_bar.set_ylim(0, agg_arr.max() * 1.4)
        ax_bar.set_title("frame attention (red = swapped)", fontsize=7)

        # Label for the row
        ax_label = fig.add_subplot(inner[:, len(frames)])
        ax_label.axis("off")
        ax_label.text(0.1, 0.5,
                      f"{name}\nswap={swap_pos}\nmodel→{answer}",
                      va="center", ha="left", fontsize=8,
                      transform=ax_label.transAxes)

    task_label = "corrupt_detect" if args.task == "corrupt" else "swap_detect"
    frame_label = "corrupted frame" if args.task == "corrupt" else "swapped frame pair"
    plt.suptitle(f"Attention rollout: {task_label} on MTL-AQA  (Qwen3-VL-8B)\n"
                 f"Red border / bar = {frame_label}", fontsize=11)
    os.makedirs("figures", exist_ok=True)
    out_path = os.path.join("figures", f"attention_rollout_{args.task}.png")
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    print(f"\n→ {out_path}")


if __name__ == "__main__":
    main()
