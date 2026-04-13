import csv
import os
import re
import random
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nobodywho import Model, Chat, Prompt, Image


def load_scenario_images(scenario_dir: str) -> list[str]:
    """Return image paths sorted numerically, skipping hidden dirs/files."""
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
    """Return a copy of the list with two random consecutive images swapped.

    Also returns the 0-based index of the first swapped image.
    """
    paths = list(image_paths)
    i = random.randrange(len(paths) - 1)
    paths[i], paths[i + 1] = paths[i + 1], paths[i]
    return paths, i


def compose_prompt(image_paths: list[str]) -> Prompt:
    return Prompt([Image(p) for p in image_paths])


def image_label(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def parse_swapped_pair(response: str) -> tuple[int, int] | tuple[None, None]:
    """Extract the last two numbers from the model response as a sorted (low, high) tuple.
    Returns (None, None) if fewer than two numbers are found."""
    numbers = re.findall(r"\d+", response)
    if len(numbers) >= 2:
        a, b = int(numbers[-2]), int(numbers[-1])
        return (min(a, b), max(a, b))
    return (None, None)


def show_result(
    scenario_name: str,
    display_paths: list[str],
    swap_index: int,
    model_response: str,
) -> None:
    n = len(display_paths)
    fig, axes = plt.subplots(
        2,
        n,
        figsize=(4 * n, 5),
        gridspec_kw={"height_ratios": [4, 1]},
    )
    fig.suptitle(scenario_name, fontsize=14, fontweight="bold")

    img_axes = axes[0] if n > 1 else [axes[0]]
    for col, (ax, path) in enumerate(zip(img_axes, display_paths)):
        img = mpimg.imread(path)
        ax.imshow(img)
        label = image_label(path)
        swapped = col in (swap_index, swap_index + 1)
        color = "red" if swapped else "white"
        ax.set_title(
            label, fontsize=10, color=color, fontweight="bold" if swapped else "normal"
        )
        ax.axis("off")
        if swapped:
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_edgecolor("red")
                spine.set_linewidth(3)

    # Merge bottom row into a single text area
    text_axes = axes[1] if n > 1 else [axes[1]]
    for ax in text_axes:
        ax.remove()
    text_ax = fig.add_subplot(2, 1, 2)
    text_ax.axis("off")

    gt_a = image_label(display_paths[swap_index])
    gt_b = image_label(display_paths[swap_index + 1])
    gt_lo, gt_hi = sorted([int(gt_a), int(gt_b)])
    pred_lo, pred_hi = parse_swapped_pair(model_response)
    ground_truth = f"Ground truth: {gt_lo} & {gt_hi}"
    model_guess = f"{pred_lo} & {pred_hi}" if pred_lo is not None else "?"
    text = f"{ground_truth}          Model: {model_guess}"
    text_ax.text(
        0.5,
        0.5,
        text,
        ha="center",
        va="center",
        wrap=True,
        fontsize=9,
        transform=text_ax.transAxes,
    )

    plt.tight_layout()
    plt.show()


EVALUATE = True
EVAL_SEED = 42


def main():
    model = Model(
        "./models/Qwen3.5-9B-Q4_K_M.gguf",
        use_gpu_if_available=True,
        image_model_path="./models/mmproj-F16.gguf",
    )

    system_prompt = (
        "You are given a sequence of images where exactly two consecutive images "
        "have been swapped. Your task is to identify which two consecutive images "
        "are out of order. Reply with only the position numbers of the two swapped "
        "images, e.g. '2 and 3'."
    )
    dataset = "./datasets/single_image_cmc"

    all_scenario_dirs = [
        os.path.join(dataset, name)
        for name in sorted(os.listdir(dataset))
        if os.path.isdir(os.path.join(dataset, name))
    ]
    valid_scenarios = [
        d for d in all_scenario_dirs if len(load_scenario_images(d)) >= 2
    ]

    if EVALUATE:
        rng = random.Random(EVAL_SEED)
        scenarios = rng.sample(valid_scenarios, min(100, len(valid_scenarios)))
    else:
        scenarios = valid_scenarios

    correct = 0
    rows: list[dict] = []
    for i, scenario_dir in enumerate(scenarios):
        scenario_name = os.path.basename(scenario_dir)
        image_paths = load_scenario_images(scenario_dir)
        display_paths, swap_index = swap_consecutive(image_paths)

        gt_a = image_label(display_paths[swap_index])
        gt_b = image_label(display_paths[swap_index + 1])
        gt_lo, gt_hi = sorted([int(gt_a), int(gt_b)])

        response: str | None = None
        try:
            chat = Chat(model, n_ctx=9124)
            chat.reset(system_prompt, tools=[])
            response = chat.ask(compose_prompt(display_paths)).completed()
            pred_lo, pred_hi = parse_swapped_pair(response)
            match = (pred_lo, pred_hi) == (gt_lo, gt_hi)
            model_str = f"{pred_lo} & {pred_hi}" if pred_lo is not None else "?"
            print(
                f"[{i + 1}/{len(scenarios)}] {scenario_name} (len={len(image_paths)}) | gt: {gt_lo} & {gt_hi} | model: {model_str} | {'OK' if match else 'WRONG'}"
            )
        except Exception as e:
            pred_lo, pred_hi = 0, 0
            match = False
            print(
                f"[{i + 1}/{len(scenarios)}] {scenario_name} (len={len(image_paths)}) | ERROR: {e}"
            )

        if EVALUATE:
            correct += match
            rows.append(
                {
                    "model_prediction_first": pred_lo,
                    "model_prediction_second": pred_hi,
                    "ground_truth_first": gt_lo,
                    "ground_truth_second": gt_hi,
                    "prediction_series_length": len(image_paths),
                }
            )
        elif response is not None:
            show_result(scenario_name, display_paths, swap_index, response)

    if EVALUATE:
        n = len(scenarios)
        print(f"\nExact match: {correct}/{n} ({100 * correct / n:.1f}%)")
        csv_path = "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions to {csv_path}")


if __name__ == "__main__":
    main()
