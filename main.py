import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from nobodywho import Model, Chat, Prompt, Image


def load_scenario_images(scenario_dir: str) -> list[str]:
    """Return image paths in the scenario dir, sorted numerically, skipping checkpoints."""
    entries = os.listdir(scenario_dir)
    paths = []
    for name in entries:
        if name.startswith("."):
            continue
        if not name.lower().endswith(".png"):
            continue
        paths.append(os.path.join(scenario_dir, name))
    paths.sort(key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    return paths


def compose_scenario_prompt(image_paths: list[str]) -> Prompt:
    return Prompt([Image(p) for p in image_paths])


def show_sequence(scenario_name: str, image_paths: list[str], description: str) -> None:
    n = len(image_paths)
    fig, axes = plt.subplots(
        2, n,
        figsize=(4 * n, 5),
        gridspec_kw={"height_ratios": [4, 1]},
    )
    fig.suptitle(scenario_name, fontsize=14, fontweight="bold")

    # Top row: images
    img_axes = axes[0] if n > 1 else [axes[0]]
    for ax, path in zip(img_axes, image_paths):
        img = mpimg.imread(path)
        ax.imshow(img)
        ax.set_title(os.path.splitext(os.path.basename(path))[0], fontsize=10)
        ax.axis("off")

    # Bottom row: description spanning all columns
    text_axes = axes[1] if n > 1 else [axes[1]]
    for ax in text_axes:
        ax.axis("off")

    # Use the first bottom cell but span the full width via a merged axis
    for ax in text_axes:
        ax.remove()
    text_ax = fig.add_subplot(2, 1, 2)
    text_ax.axis("off")
    text_ax.text(
        0.5, 0.5,
        description,
        ha="center", va="center",
        wrap=True,
        fontsize=9,
        transform=text_ax.transAxes,
    )

    plt.tight_layout()
    plt.show()


def main():
    model = Model(
        "./models/Qwen3.5-9B-Q4_K_M.gguf",
        use_gpu_if_available=True,
        image_model_path="./models/mmproj-F16.gguf",
    )
    chat = Chat(model, n_ctx=8192)

    system_prompt = (
        "Describe step by step, what you see in the images. "
        "Make it ideally one sentence, coherent with the ordering of the images. "
        "Put a number like (1), (2), ... to the sentence to know which image is it."
    )
    dataset = "./datasets/single_image_cmc"

    scenario_dirs = sorted(os.listdir(dataset))
    for scenario_name in scenario_dirs:
        scenario_dir = os.path.join(dataset, scenario_name)
        if not os.path.isdir(scenario_dir):
            continue

        image_paths = load_scenario_images(scenario_dir)
        if not image_paths:
            continue

        chat.reset(system_prompt, tools=[])
        prompt = compose_scenario_prompt(image_paths)
        print(f"-- Answering: {scenario_name} --")
        description = chat.ask(prompt).completed()
        print(description)

        show_sequence(scenario_name, image_paths, description)


if __name__ == "__main__":
    main()
