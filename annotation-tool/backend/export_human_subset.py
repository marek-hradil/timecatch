#!/usr/bin/env python3
"""
Export the exact image sequences that approved Prolific participants saw,
so AI models can be evaluated on the same stimuli.

Reads Firestore (needs internet), replays the deterministic build_plan() to
recover frame order (including the swap position for detect-task swapped items),
and writes one JSON manifest per (dataset, task) to human_subset/.

Prerequisites:
  1. Run sync_prolific.py first — participants need prolific_status in Firestore.
  2. datasets/<DS>/scenes_filtered.json must exist locally (only the manifest,
     not the actual images).

Usage:
  cd annotation-tool
  .venv/bin/python -m backend.export_human_subset [--out-dir <path>]
"""
import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

# --- path setup -----------------------------------------------------------
# Add annotation-tool/ so `backend` is importable as a package (required for
# relative imports inside study.py / sampling.py)
_BACKEND_DIR = Path(__file__).parent
_ANNOTATION_TOOL_DIR = _BACKEND_DIR.parent
_REPO_ROOT = _ANNOTATION_TOOL_DIR.parent
sys.path.insert(0, str(_ANNOTATION_TOOL_DIR))
sys.path.insert(0, str(_REPO_ROOT))  # for `from dataset import Dataset` in sampling.py

from google.cloud import firestore
from backend.study import build_plan, ground_truth_for  # noqa: E402

# Annotation-tool dataset key → lowercase name used in filenames
_DS_FILENAME = {
    "CRAFT":   "craft",
    "CLEVRER": "clevrer",
    "MTL-AQA": "mtl-aqa",
    "drive_lm": "drive-lm",
}


def _shown_order(frame_urls: list[str]) -> list[int]:
    """Extract original frame indices from frame URLs (stem = original index)."""
    return [int(Path(url).stem) for url in frame_urls]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default=str(_REPO_ROOT / "human_subset"),
                        help="Directory to write manifest JSONs (default: <repo>/human_subset)")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    db = firestore.Client()

    # -----------------------------------------------------------------------
    # 1. Load all approved participants
    # -----------------------------------------------------------------------
    print("Loading approved participants from Firestore...")
    all_participants = list(db.collection("participants").stream())
    approved = [
        p.to_dict() for p in all_participants
        if p.to_dict().get("prolific_status") == "APPROVED"
        and p.to_dict().get("scene_id", "main") == "main"
    ]
    print(f"  {len(all_participants)} total, {len(approved)} approved")

    if not approved:
        print("No approved participants found. Run sync_prolific.py first.")
        sys.exit(1)

    # -----------------------------------------------------------------------
    # 2. Load annotations for approved participants
    # -----------------------------------------------------------------------
    print("Loading annotations from Firestore...")
    all_anns = list(db.collection("annotations").stream())
    approved_pids = {p["prolific_pid"] for p in approved}

    # index: (pid, dataset, task, scene_id) → {sample_index → annotation_doc}
    ann_index: dict[tuple, dict[int, dict]] = defaultdict(dict)
    for doc in all_anns:
        d = doc.to_dict()
        pid = d.get("prolific_pid")
        if pid not in approved_pids:
            continue
        key = (pid, d.get("dataset"), d.get("task"), d.get("scene_id", "main"))
        ann_index[key][d.get("sample_index")] = d

    # -----------------------------------------------------------------------
    # 3. Reconstruct plans and build manifests
    # -----------------------------------------------------------------------
    # manifests[(dataset, task)] = dict mapping (scene_name, shown_order_tuple)
    #                               → stimulus entry dict
    manifests: dict[tuple, dict] = defaultdict(dict)
    skipped = 0

    for p in approved:
        pid     = p["prolific_pid"]
        dataset = p.get("dataset")
        task    = p.get("task")
        scene_id = p.get("scene_id", "main")

        if not dataset or not task:
            print(f"  WARNING: participant {pid} missing dataset/task, skipping")
            skipped += 1
            continue

        # Replay exact sample plan
        try:
            plan = build_plan(pid, dataset, task)
            gt_list = ground_truth_for(pid, dataset, task)
        except Exception as e:
            print(f"  WARNING: build_plan failed for {pid} ({dataset}/{task}): {e}")
            skipped += 1
            continue

        anns_for_p = ann_index.get((pid, dataset, task, scene_id), {})
        if not anns_for_p:
            print(f"  WARNING: no annotations found for {pid} ({dataset}/{task}), skipping")
            skipped += 1
            continue

        participant_ok = True
        for plan_item, gt_item in zip(plan, gt_list):
            i = plan_item.sample_index
            if i not in anns_for_p:
                continue
            stored_ann = anns_for_p[i]
            stored_name = stored_ann.get("sample_name")
            if stored_name and stored_name != plan_item.sample_name:
                print(
                    f"  WARNING: sample_name mismatch for {pid} index {i}: "
                    f"stored={stored_name!r} reconstructed={plan_item.sample_name!r}. "
                    f"Skipping participant."
                )
                participant_ok = False
                break

        if not participant_ok:
            skipped += 1
            continue

        # Collect stimuli
        for plan_item, gt_item in zip(plan, gt_list):
            if plan_item.is_attention_check:
                continue  # exclude attention checks

            i = plan_item.sample_index
            if i not in anns_for_p:
                continue

            stored_ann = anns_for_p[i]
            shown_order = _shown_order(plan_item.frame_urls)
            stimulus_key = (plan_item.sample_name, tuple(shown_order))

            # ground_truth: bool for detect, list[int, int] for localize
            gt_raw = gt_item.ground_truth
            if isinstance(gt_raw, tuple):
                gt_json = list(gt_raw)
            else:
                gt_json = gt_raw  # bool

            human_answer = stored_ann.get("human_answer")
            if isinstance(human_answer, list):
                human_answer = human_answer  # already list — fine for JSON

            manifest_key = (dataset, task)
            if stimulus_key not in manifests[manifest_key]:
                manifests[manifest_key][stimulus_key] = {
                    "scene_name":   plan_item.sample_name,
                    "shown_order":  shown_order,
                    "ground_truth": gt_json,
                    "n_frames":     plan_item.n_frames,
                    "annotations":  [],
                }

            manifests[manifest_key][stimulus_key]["annotations"].append({
                "pid":          pid,
                "human_answer": human_answer,
            })

    # -----------------------------------------------------------------------
    # 4. Write manifest JSONs
    # -----------------------------------------------------------------------
    total_stimuli = 0
    total_annotations = 0
    for (dataset, task), stimuli_dict in sorted(manifests.items()):
        stimuli = list(stimuli_dict.values())
        filename = f"swap_{task}_{_DS_FILENAME.get(dataset, dataset.lower())}.json"
        out_path = out_dir / filename
        payload = {
            "dataset": dataset,
            "task": task,
            "stimuli": stimuli,
        }
        with open(out_path, "w") as f:
            json.dump(payload, f, indent=2)
        n_ann = sum(len(s["annotations"]) for s in stimuli)
        print(f"  Wrote {out_path}: {len(stimuli)} stimuli, {n_ann} annotations")
        total_stimuli += len(stimuli)
        total_annotations += n_ann

    print(f"\nDone. {total_stimuli} stimuli, {total_annotations} annotations total. "
          f"{skipped} participants skipped.")


if __name__ == "__main__":
    main()
