#!/usr/bin/env python3
"""
Usage:
    python3 check_participant.py <prolific_pid> [scene_id]

scene_id defaults to "main".
"""
import sys
from google.cloud import firestore

def main():
    if len(sys.argv) < 2:
        print("Usage: python3 check_participant.py <prolific_pid> [scene_id]")
        sys.exit(1)

    pid = sys.argv[1]
    scene_id = sys.argv[2] if len(sys.argv) > 2 else "main"

    db = firestore.Client()

    # Find participant doc
    docs = list(db.collection("participants")
                  .where("prolific_pid", "==", pid)
                  .where("scene_id", "==", scene_id)
                  .stream())

    if not docs:
        print(f"No participant found for pid={pid} scene_id={scene_id}")
        sys.exit(1)

    p = docs[0].to_dict()
    dataset = p.get("dataset")
    task = p.get("task")
    created = p.get("created_at", "?")[:19]
    completed = (p.get("completed_at") or "incomplete")[:19]
    instr_dur = p.get("instructions_duration_s")
    video_played = p.get("video_played")
    lightbox = p.get("instructions_lightbox_opens")

    print(f"\nParticipant: {pid}")
    print(f"Dataset/Task: {dataset} / {task}  (scene_id={scene_id})")
    print(f"Created: {created}  Completed: {completed}")
    print(f"Instructions: {instr_dur:.1f}s  |  video_played={video_played}  |  lightbox_opens={lightbox}")

    # Fetch annotations
    anns = list(db.collection("annotations")
                  .where("prolific_pid", "==", pid)
                  .where("dataset", "==", dataset)
                  .where("task", "==", task)
                  .where("scene_id", "==", scene_id)
                  .stream())

    anns = sorted([a.to_dict() for a in anns], key=lambda d: d.get("sample_index", 0))

    if not anns:
        print("No annotations found.")
        return

    correct = 0
    total = len(anns)
    attn_passed = 0
    attn_failed = 0
    attn_details = []

    print(f"\n{'#':>3}  {'sample':<20}  {'answer':<12}  {'ground_truth':<12}  {'OK':>4}  {'attn':>5}  {'dur':>6}  {'opens':>5}")
    print("-" * 85)

    for d in anns:
        idx = d.get("sample_index")
        sample = d.get("sample_name", "?")
        ha = d.get("human_answer")
        gt = d.get("ground_truth")
        is_attn = d.get("is_attention_check", False)
        dur = d.get("view_duration_s")
        opens = d.get("viewer_opens")

        # normalise to comparable types
        if isinstance(ha, list):
            ha = tuple(ha)
        if isinstance(gt, list):
            gt = tuple(gt)

        ok = ha == gt
        if ok:
            correct += 1
        mark = "✓" if ok else "✗"

        attn_mark = ""
        if is_attn:
            if ok:
                attn_passed += 1
                attn_mark = "PASS"
            else:
                attn_failed += 1
                attn_mark = "FAIL"
            attn_details.append((idx, ok))

        dur_str = f"{dur:.1f}s" if dur is not None else "?"
        opens_str = str(opens) if opens is not None else "?"

        print(f"{idx:>3}  {str(sample):<20}  {str(ha):<12}  {str(gt):<12}  {mark:>4}  {attn_mark:>5}  {dur_str:>6}  {opens_str:>5}")

    print("-" * 85)
    print(f"\nAccuracy: {correct}/{total} ({100*correct/total:.0f}%)")

    attn_total = attn_passed + attn_failed
    if attn_total:
        print(f"Attention checks: {attn_passed}/{attn_total} passed", end="")
        if attn_failed:
            failed_idx = [str(i) for i, ok in attn_details if not ok]
            print(f"  — FAILED at samples: {', '.join(failed_idx)}", end="")
        print()
    else:
        print("Attention checks: none flagged in data")

    avg_dur = sum(d.get("view_duration_s", 0) for d in anns if d.get("view_duration_s")) / total
    total_opens = sum(d.get("viewer_opens", 0) for d in anns if d.get("viewer_opens"))
    print(f"Avg time/sample: {avg_dur:.1f}s  |  Total viewer opens: {total_opens}")

if __name__ == "__main__":
    main()
