import csv
import os
import re
import random
import tempfile
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from collections import defaultdict
from PIL import Image as PILImage
from nobodywho import Model, Chat, Prompt, Image


def load_scenario_images(scenario_dir: str) -> list[str]:
    paths = []
    for name in os.listdir(scenario_dir):
        if name.startswith("."):
            continue
        if not name.lower().endswith(".png"):
            continue
        paths.append(os.path.join(scenario_dir, name))
    paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return paths


def swap_consecutive(image_paths: list[str]) -> tuple[list[str], int]:
    paths = list(image_paths)
    i = random.randrange(len(paths) - 1)
    paths[i], paths[i + 1] = paths[i + 1], paths[i]
    return paths, i


def corrupt_random_image(image_paths: list[str]) -> tuple[list[str], int]:
    paths = list(image_paths)
    i = random.randrange(len(paths))
    with PILImage.open(paths[i]) as ref:
        w, h = ref.size
    noise = np.random.randint(0, 256, (h, w, 3), dtype=np.uint8)
    noisy_img = PILImage.fromarray(noise)
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    noisy_img.save(tmp.name)
    tmp.close()
    paths[i] = tmp.name
    return paths, i


def make_composite_image(image_paths: list[str], gap: int = 4) -> str:
    imgs = [PILImage.open(p) for p in image_paths]
    w = sum(im.width for im in imgs) + gap * (len(imgs) - 1)
    h = max(im.height for im in imgs)
    composite = PILImage.new("RGB", (w, h), "white")
    x = 0
    for im in imgs:
        composite.paste(im, (x, 0))
        x += im.width + gap
    tmp = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
    composite.save(tmp.name)
    tmp.close()
    return tmp.name


def compose_prompt_series(image_paths: list[str]) -> Prompt:
    return Prompt([Image(p) for p in image_paths])


def compose_prompt_composite(composite_path: str) -> Prompt:
    return Prompt([Image(composite_path)])


def image_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def parse_yes_no(response: str) -> bool | None:
    normalized = response.strip().lower()
    if normalized == "yes":
        return True
    if normalized == "no":
        return False
    has_yes = bool(re.search(r"\byes\b", normalized))
    has_no = bool(re.search(r"\bno\b", normalized))
    if has_yes and not has_no:
        print(f"  [WARN] non-standard response, inferred yes: {response!r}")
        return True
    if has_no and not has_yes:
        print(f"  [WARN] non-standard response, inferred no: {response!r}")
        return False
    print(f"  [WARN] cannot parse yes/no from: {response!r}")
    return None


def save_failure(
    scenario_name: str,
    display_paths: list[str],
    has_anomaly: bool,
    anomaly_index: int | None,
    model_response: str,
    mode: str,
    task: str = "swap",
    out_dir: str = "failures",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = len(display_paths)
    fig, axes = plt.subplots(
        2, n, figsize=(4 * n, 5), gridspec_kw={"height_ratios": [4, 1]}
    )
    suffix = "anomaly" if has_anomaly else "normal"
    fig.suptitle(
        f"{scenario_name} [{mode}] [{task}] [{suffix}]", fontsize=14, fontweight="bold"
    )

    img_axes = axes[0] if n > 1 else [axes[0]]
    for col, (ax, path) in enumerate(zip(img_axes, display_paths)):
        img = mpimg.imread(path)
        ax.imshow(img)
        label = image_label(path)
        if task == "swap":
            highlighted = (
                has_anomaly
                and anomaly_index is not None
                and col in (anomaly_index, anomaly_index + 1)
            )
        else:
            highlighted = (
                has_anomaly and anomaly_index is not None and col == anomaly_index
            )
        color = "red" if highlighted else "white"
        ax.set_title(
            label,
            fontsize=10,
            color=color,
            fontweight="bold" if highlighted else "normal",
        )
        ax.axis("off")
        if highlighted:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

    text_axes = axes[1] if n > 1 else [axes[1]]
    for ax in text_axes:
        ax.remove()
    text_ax = fig.add_subplot(2, 1, 2)
    text_ax.axis("off")

    gt_str = "yes (anomaly present)" if has_anomaly else "no (normal)"
    raw_truncated = model_response.replace("\n", " ")[:120]
    text_ax.text(
        0.5,
        0.6,
        f"Ground truth: {gt_str}          Model: {model_response.strip()!r}",
        ha="center",
        va="center",
        fontsize=9,
        transform=text_ax.transAxes,
    )
    text_ax.text(
        0.5,
        0.2,
        f'Response: "{raw_truncated}"',
        ha="center",
        va="center",
        fontsize=7,
        color="gray",
        transform=text_ax.transAxes,
        wrap=True,
    )

    plt.tight_layout()
    fig.savefig(
        os.path.join(out_dir, f"{scenario_name}_len{n}_{task}_{mode}_{suffix}.png"),
        dpi=80,
    )
    plt.close(fig)


def plot_results(rows: list[dict]) -> None:
    counts = defaultdict(
        lambda: defaultdict(lambda: {"TP": 0, "TN": 0, "FP": 0, "FN": 0})
    )
    for row in rows:
        if row["outcome"] == "INVALID":
            continue
        counts[row["input_mode"]][row["sequence_length"]][row["outcome"]] += 1

    outcome_colors = {
        "TP": "#4caf50",
        "TN": "#2196f3",
        "FP": "#ff9800",
        "FN": "#f44336",
    }
    outcome_order = ["TP", "TN", "FP", "FN"]
    modes = sorted(counts.keys())

    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 5))
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        lengths = sorted(counts[mode].keys())
        x = list(range(len(lengths)))
        bottoms = [0] * len(lengths)

        for outcome in outcome_order:
            vals = [counts[mode][l][outcome] for l in lengths]
            ax.bar(
                x, vals, bottom=bottoms, color=outcome_colors[outcome], label=outcome
            )
            bottoms = [b + v for b, v in zip(bottoms, vals)]

        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in lengths])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Count")
        ax.set_title(f"Mode: {mode}")
        ax.legend()

    fig.suptitle(
        "Outcome distribution per sequence length", fontsize=14, fontweight="bold"
    )
    plt.tight_layout()
    plt.savefig("results_stacked_bar.png", dpi=100)
    plt.show()


EVAL_SEED = 42

# Set to "swap", "corrupt", or "localize".
# "localize": always corrupts one image with Gaussian noise and asks the model which frame index is corrupted.
TASK = "localize"

SYSTEM_PROMPT_SERIES = (
    "You are given a sequence of images. Each image shows a colored dot at a position along a curved path. "
    "The dot moves smoothly along the path from frame to frame. Two consecutive images may or may not have been swapped. "
    "Your task is to determine whether the sequence contains a swap. "
    "Reply with only 'yes' if two consecutive images are out of order, or 'no' if the sequence is in the correct order."
)

SYSTEM_PROMPT_COMPOSITE = (
    "You are given a single image showing a sequence of frames arranged left to right. "
    "Each frame shows a colored dot at a position along a curved path. "
    "The dot moves smoothly along the path from frame to frame. Two consecutive frames may or may not have been swapped. "
    "Your task is to determine whether the sequence contains a swap. "
    "Reply with only 'yes' if two consecutive frames are out of order, or 'no' if the sequence is in the correct order."
)

SYSTEM_PROMPT_SERIES_CORRUPT = (
    "You are given a sequence of images. Each image shows a colored dot at a position along a curved path. "
    "The dot moves smoothly along the path from frame to frame. One image in the sequence may or may not have been replaced with random noise. "
    "Your task is to determine whether the sequence contains a corrupted image. "
    "Reply with only 'yes' if one image appears to be random noise, or 'no' if the sequence looks normal."
)

SYSTEM_PROMPT_COMPOSITE_CORRUPT = (
    "You are given a single image showing a sequence of frames arranged left to right. "
    "Each frame shows a colored dot at a position along a curved path. "
    "The dot moves smoothly along the path from frame to frame. One frame in the sequence may or may not have been replaced with random noise. "
    "Your task is to determine whether the sequence contains a corrupted frame. "
    "Reply with only 'yes' if one frame appears to be random noise, or 'no' if the sequence looks normal."
)

SYSTEM_PROMPT_SERIES_LOCALIZE = (
    "You are given a sequence of images. Each image shows a colored dot at a position along a curved path. "
    "The dot moves smoothly along the path from frame to frame. Exactly one image has been replaced with random Gaussian noise. "
    "Reply with only the number of the corrupted image (1 for the first image, 2 for the second, etc.)."
)

SYSTEM_PROMPT_COMPOSITE_LOCALIZE = (
    "You are given a single image showing a sequence of frames arranged left to right. "
    "Each frame shows a colored dot at a position along a curved path. "
    "The dot moves smoothly along the path from frame to frame. Exactly one frame has been replaced with random Gaussian noise. "
    "Reply with only the number of the corrupted frame (1 for the leftmost frame, 2 for the second frame, etc.)."
)


def plot_localize_results(rows: list[dict]) -> None:
    counts = defaultdict(lambda: defaultdict(lambda: {"correct": 0, "wrong": 0, "INVALID": 0}))
    for row in rows:
        counts[row["input_mode"]][row["sequence_length"]][row["outcome"]] += 1

    outcome_colors = {"correct": "#4caf50", "wrong": "#f44336", "INVALID": "#9e9e9e"}
    outcome_order = ["correct", "wrong", "INVALID"]
    modes = sorted(counts.keys())

    fig, axes = plt.subplots(1, len(modes), figsize=(7 * len(modes), 5))
    if len(modes) == 1:
        axes = [axes]

    for ax, mode in zip(axes, modes):
        lengths = sorted(counts[mode].keys())
        x = list(range(len(lengths)))
        bottoms = [0] * len(lengths)
        for outcome in outcome_order:
            vals = [counts[mode][l][outcome] for l in lengths]
            ax.bar(x, vals, bottom=bottoms, color=outcome_colors[outcome], label=outcome)
            bottoms = [b + v for b, v in zip(bottoms, vals)]
        ax.set_xticks(x)
        ax.set_xticklabels([str(l) for l in lengths])
        ax.set_xlabel("Sequence length")
        ax.set_ylabel("Count")
        ax.set_title(f"Mode: {mode}")
        ax.legend()

    fig.suptitle("Localization accuracy per sequence length", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig("results_stacked_bar.png", dpi=100)
    plt.show()


def parse_frame_number(response: str, seq_len: int) -> int | None:
    match = re.search(r"\b(\d+)\b", response.strip())
    if match:
        n = int(match.group(1))
        if 1 <= n <= seq_len:
            return n
        print(f"  [WARN] frame number {n} out of range [1, {seq_len}]: {response!r}")
        return None
    print(f"  [WARN] cannot parse frame number from: {response!r}")
    return None


def run_mode(model, system_prompt: str, prompt: Prompt, expected_pattern: str = r"\s*(yes|no)\s*") -> str:
    chat = Chat(model, n_ctx=9124)
    chat.reset(system_prompt, tools=[])
    response = chat.ask(prompt).completed()
    if not re.fullmatch(expected_pattern, response, re.IGNORECASE):
        print(f"  [WARN] unexpected format: {response!r}")
    return response


def main():
    model = Model(
        "./models/Qwen3.5-9B-Q4_K_M.gguf",
        use_gpu_if_available=True,
        projection_model_path="./models/Qwen3.5-9B-Q4_K_M-mmproj-F16.gguf",
    )

    dataset = "./datasets/single_image_abstract_ball"

    all_scenario_dirs = [
        os.path.join(dataset, name)
        for name in sorted(os.listdir(dataset))
        if os.path.isdir(os.path.join(dataset, name))
    ]
    valid_scenarios = [
        d for d in all_scenario_dirs if len(load_scenario_images(d)) >= 2
    ]

    rng = random.Random(EVAL_SEED)
    scenarios = rng.sample(valid_scenarios, min(100, len(valid_scenarios)))

    correct: dict[str, int] = {"series": 0, "composite": 0}
    total: dict[str, int] = {"series": 0, "composite": 0}
    rows: list[dict] = []

    if TASK == "swap":
        series_sys = SYSTEM_PROMPT_SERIES
        composite_sys = SYSTEM_PROMPT_COMPOSITE
    elif TASK == "localize":
        series_sys = SYSTEM_PROMPT_SERIES_LOCALIZE
        composite_sys = SYSTEM_PROMPT_COMPOSITE_LOCALIZE
    else:
        series_sys = SYSTEM_PROMPT_SERIES_CORRUPT
        composite_sys = SYSTEM_PROMPT_COMPOSITE_CORRUPT

    total_trials = len(scenarios) * (1 if TASK == "localize" else 2)
    trial_num = 0

    for scenario_dir in scenarios:
        scenario_name = os.path.basename(scenario_dir)
        image_paths = load_scenario_images(scenario_dir)
        seq_len = len(image_paths)

        corrupt_tmp = None
        if TASK == "swap":
            modified_paths, anomaly_index = swap_consecutive(image_paths)
        else:
            modified_paths, anomaly_index = corrupt_random_image(image_paths)
            corrupt_tmp = modified_paths[anomaly_index]

        if TASK == "localize":
            trials = [(True, modified_paths, anomaly_index)]
        else:
            trials = [
                (True, modified_paths, anomaly_index),
                (False, list(image_paths), None),
            ]

        for has_anomaly, display_paths, anomaly_index_or_none in trials:
            trial_num += 1
            composite_path = make_composite_image(display_paths)

            modes_data = [
                ("series", series_sys, compose_prompt_series(display_paths)),
                ("composite", composite_sys, compose_prompt_composite(composite_path)),
            ]

            try:
                for mode, system_prompt, prompt in modes_data:
                    response: str | None = None
                    outcome = "INVALID"
                    is_correct = False
                    model_answer = None
                    try:
                        if TASK == "localize":
                            response = run_mode(model, system_prompt, prompt, r"\s*\d+\s*")
                            frame_num = parse_frame_number(response, seq_len)
                            model_answer = frame_num
                            if frame_num is not None:
                                outcome = "correct" if (frame_num - 1) == anomaly_index_or_none else "wrong"
                                is_correct = outcome == "correct"
                            print(
                                f"[{trial_num}/{total_trials}] {scenario_name} [{mode}] "
                                f"(len={seq_len}) "
                                f"| gt: {anomaly_index_or_none + 1} "
                                f"| model: {frame_num} "
                                f"| {outcome}"
                            )
                        else:
                            response = run_mode(model, system_prompt, prompt)
                            model_bool = parse_yes_no(response)
                            model_answer = model_bool
                            if model_bool is not None:
                                if has_anomaly and model_bool:
                                    outcome = "TP"
                                elif has_anomaly and not model_bool:
                                    outcome = "FN"
                                elif not has_anomaly and model_bool:
                                    outcome = "FP"
                                else:
                                    outcome = "TN"
                                is_correct = outcome in ("TP", "TN")
                            print(
                                f"[{trial_num}/{total_trials}] {scenario_name} [{mode}] "
                                f"(len={seq_len}, anomaly={has_anomaly}) "
                                f"| gt: {'yes' if has_anomaly else 'no'} "
                                f"| model: {model_bool} "
                                f"| {outcome}"
                            )
                    except Exception as e:
                        print(
                            f"[{trial_num}/{total_trials}] {scenario_name} [{mode}] "
                            f"(len={seq_len}) | ERROR: {e}"
                        )

                    if outcome != "INVALID":
                        correct[mode] += is_correct
                        total[mode] += 1
                    if outcome in ("FP", "FN", "wrong") and response is not None:
                        save_failure(
                            scenario_name,
                            display_paths,
                            has_anomaly,
                            anomaly_index_or_none,
                            response,
                            mode,
                            TASK,
                        )
                    rows.append(
                        {
                            "task": TASK,
                            "input_mode": mode,
                            "scenario": scenario_name,
                            "sequence_length": seq_len,
                            "has_anomaly": has_anomaly,
                            "model_answer": model_answer,
                            "outcome": outcome,
                        }
                    )
            finally:
                os.unlink(composite_path)

        if corrupt_tmp is not None:
            os.unlink(corrupt_tmp)

    for mode in ("series", "composite"):
        c = correct[mode]
        n = total[mode]
        if n > 0:
            print(f"\n[{mode}] Accuracy: {c}/{n} ({100 * c / n:.1f}%)")

    if rows:
        csv_path = "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions to {csv_path}")
        if TASK == "localize":
            plot_localize_results(rows)
        else:
            plot_results(rows)


if __name__ == "__main__":
    main()
