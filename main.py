import csv
import os
import re
import random
import tempfile
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
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


def parse_swapped_pair(response: str) -> tuple[int, int] | tuple[None, None]:
    m = re.fullmatch(r"\s*(\d+)\s+(\d+)\s*", response)
    if m:
        a, b = int(m.group(1)), int(m.group(2))
        return (min(a, b), max(a, b))
    # fallback: last two numbers anywhere in the response
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
        2, n, figsize=(4 * n, 5), gridspec_kw={"height_ratios": [4, 1]}
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

    text_axes = axes[1] if n > 1 else [axes[1]]
    for ax in text_axes:
        ax.remove()
    text_ax = fig.add_subplot(2, 1, 2)
    text_ax.axis("off")

    gt_a = image_label(display_paths[swap_index])
    gt_b = image_label(display_paths[swap_index + 1])
    gt_lo, gt_hi = sorted([int(gt_a), int(gt_b)])
    pred_lo, pred_hi = parse_swapped_pair(model_response)
    model_guess = f"{pred_lo} & {pred_hi}" if pred_lo is not None else "?"
    text_ax.text(
        0.5,
        0.5,
        f"Ground truth: {gt_lo} & {gt_hi}          Model: {model_guess}",
        ha="center",
        va="center",
        wrap=True,
        fontsize=9,
        transform=text_ax.transAxes,
    )
    plt.tight_layout()
    plt.show()


def save_failure(
    scenario_name: str,
    display_paths: list[str],
    swap_index: int,
    model_response: str,
    mode: str,
    out_dir: str = "failures",
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    n = len(display_paths)
    fig, axes = plt.subplots(
        2, n, figsize=(4 * n, 5), gridspec_kw={"height_ratios": [4, 1]}
    )
    fig.suptitle(f"{scenario_name} [{mode}]", fontsize=14, fontweight="bold")

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

    text_axes = axes[1] if n > 1 else [axes[1]]
    for ax in text_axes:
        ax.remove()
    text_ax = fig.add_subplot(2, 1, 2)
    text_ax.axis("off")

    gt_a = image_label(display_paths[swap_index])
    gt_b = image_label(display_paths[swap_index + 1])
    gt_lo, gt_hi = sorted([int(gt_a), int(gt_b)])
    pred_lo, pred_hi = parse_swapped_pair(model_response)
    model_guess = f"{pred_lo} & {pred_hi}" if pred_lo is not None else "?"
    raw_truncated = model_response.replace("\n", " ")[:120]
    text_ax.text(
        0.5,
        0.6,
        f"Ground truth: {gt_lo} & {gt_hi}          Model: {model_guess}",
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
    fig.savefig(os.path.join(out_dir, f"{scenario_name}_len{n}_{mode}.png"), dpi=80)
    plt.close(fig)


EVALUATE = True
EVAL_SEED = 42

SYSTEM_PROMPT_SERIES = (
    "You are given a sequence of images. Each image shows a colored dot at a position along a curved path. "
    "The dot moves smoothly along the path from frame to frame. Exactly two consecutive images have been swapped. "
    "Your task is to identify which two consecutive images are out of order. "
    "Reply with only the 0-based index numbers of the two swapped images in the format: x y"
)

SYSTEM_PROMPT_COMPOSITE = (
    "You are given a single image showing a sequence of frames arranged left to right. "
    "Each frame shows a colored dot at a position along a curved path. "
    "The dot moves smoothly along the path from frame to frame. Exactly two consecutive frames have been swapped. "
    "Your task is to identify which two consecutive frames are out of order. "
    "Reply with only the 0-based index numbers of the two swapped frames in the format: x y"
)


def run_mode(model, system_prompt: str, prompt: Prompt) -> str:
    chat = Chat(model, n_ctx=9124)
    chat.reset(system_prompt, tools=[])
    response = chat.ask(prompt).completed()
    if not re.fullmatch(r"\s*\d+\s+\d+\s*", response):
        print(f"  [WARN] unexpected format: {response!r}")
    return response


def main():
    model = Model(
        "./models/InternVL3.gguf",
        use_gpu_if_available=True,
        projection_model_path="./models/InternVL3-mmproj.gguf",
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

    if EVALUATE:
        rng = random.Random(EVAL_SEED)
        scenarios = rng.sample(valid_scenarios, min(100, len(valid_scenarios)))
    else:
        scenarios = valid_scenarios

    correct: dict[str, int] = {"series": 0, "composite": 0}
    rows: list[dict] = []

    for i, scenario_dir in enumerate(scenarios):
        scenario_name = os.path.basename(scenario_dir)
        image_paths = load_scenario_images(scenario_dir)
        display_paths, swap_index = swap_consecutive(image_paths)

        gt_a = image_label(display_paths[swap_index])
        gt_b = image_label(display_paths[swap_index + 1])
        gt_lo, gt_hi = sorted([int(gt_a), int(gt_b)])

        composite_path = make_composite_image(display_paths)

        modes = [
            ("series", SYSTEM_PROMPT_SERIES, compose_prompt_series(display_paths)),
            (
                "composite",
                SYSTEM_PROMPT_COMPOSITE,
                compose_prompt_composite(composite_path),
            ),
        ]

        try:
            for mode, system_prompt, prompt in modes:
                response: str | None = None
                pred_lo, pred_hi = None, None
                match = False
                try:
                    response = run_mode(model, system_prompt, prompt)
                    pred_lo, pred_hi = parse_swapped_pair(response)
                    match = (pred_lo, pred_hi) == (gt_lo, gt_hi)
                    model_str = f"{pred_lo} & {pred_hi}" if pred_lo is not None else "?"
                    print(
                        f"[{i + 1}/{len(scenarios)}] {scenario_name} [{mode}] (len={len(image_paths)}) "
                        f"| gt: {gt_lo} & {gt_hi} | model: {model_str} | {'OK' if match else 'WRONG'}"
                    )
                except Exception as e:
                    print(
                        f"[{i + 1}/{len(scenarios)}] {scenario_name} [{mode}] (len={len(image_paths)}) | ERROR: {e}"
                    )

                if EVALUATE:
                    correct[mode] += match
                    if not match and response is not None:
                        save_failure(
                            scenario_name, display_paths, swap_index, response, mode
                        )
                    rows.append(
                        {
                            "input_mode": mode,
                            "model_prediction_first": pred_lo,
                            "model_prediction_second": pred_hi,
                            "ground_truth_first": gt_lo,
                            "ground_truth_second": gt_hi,
                            "prediction_series_length": len(image_paths),
                        }
                    )
                elif response is not None and mode == "series":
                    show_result(scenario_name, display_paths, swap_index, response)
        finally:
            os.unlink(composite_path)

    if EVALUATE:
        n = len(scenarios)
        for mode in ("series", "composite"):
            c = correct[mode]
            print(f"\n[{mode}] Exact match: {c}/{n} ({100 * c / n:.1f}%)")
        csv_path = "results.csv"
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        print(f"Saved predictions to {csv_path}")


if __name__ == "__main__":
    main()
