import csv
import os
from datetime import datetime


class Results:
    def __init__(
        self,
        experiment: str,
        model_id: str,
        total: int | None = None,
        out_dir: str = "outputs",
    ):
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_dir = os.path.join(out_dir, f"{experiment}_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)

        self._csv_path = os.path.join(run_dir, "results.csv")
        self._experiment = experiment
        self._model_id = model_id
        self._total = total
        self._count = 0

        with open(self._csv_path, "w", newline="") as f:
            csv.writer(f).writerow(["scenario", "seq_len", "ground_truth", "answer"])

    def log(
        self,
        scenario: str,
        seq_len: int,
        ground_truth: object,
        answer: object,
    ) -> None:
        self._count += 1

        counter = f"[{self._count}/{self._total}]" if self._total else f"[{self._count}]"
        print(f"{counter} {scenario}  seq={seq_len}  gt={ground_truth}  answer={answer}", flush=True)

        with open(self._csv_path, "a", newline="") as f:
            csv.writer(f).writerow([scenario, seq_len, ground_truth, answer])

    def summary(self) -> None:
        print(f"\n=== {self._experiment} | {self._model_id} | {self._count} trials ===", flush=True)
