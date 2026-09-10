#!/usr/bin/env python3
"""
Print per-participant accuracy for all approved submissions in the project.

Usage:
    python3 study_results.py --token <prolific_api_token>
    python3 study_results.py --token <prolific_api_token> --study-id <id>

Get your API token at: https://app.prolific.com/researcher/settings/api
"""
import argparse
from datetime import datetime
import requests
from google.cloud import firestore
from google.cloud.firestore_v1 import FieldFilter

PROLIFIC_API = "https://api.prolific.com/api/v1"
PROJECT_ID = "6a1ec42d940f1321969c30d0"


def prolific_get(path: str, token: str) -> dict:
    resp = requests.get(
        f"{PROLIFIC_API}{path}",
        headers={"Authorization": f"Token {token}"},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()


def fetch_all_studies(token: str) -> list[dict]:
    try:
        data = prolific_get(f"/projects/{PROJECT_ID}/studies/", token)
        results = data.get("results", [])
        if results or "results" in data:
            return results
    except requests.HTTPError:
        pass
    data = prolific_get("/studies/", token)
    return [s for s in data.get("results", []) if s.get("project") == PROJECT_ID]


def fetch_submissions(study_id: str, token: str) -> list[dict]:
    results = []
    path = f"/studies/{study_id}/submissions/?limit=100"
    while path:
        data = prolific_get(path, token)
        results.extend(data.get("results", []))
        next_url = data.get("next")
        path = next_url.replace(PROLIFIC_API, "") if next_url else None
    return results


def compute_accuracy(anns: list[dict]) -> tuple[int, int]:
    correct = 0
    for d in anns:
        ha = d.get("human_answer")
        gt = d.get("ground_truth")
        if ha is None or gt is None:
            continue
        if isinstance(ha, list):
            ha = tuple(ha)
        if isinstance(gt, list):
            gt = tuple(gt)
        if ha == gt:
            correct += 1
    return correct, len(anns)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Prolific API token")
    parser.add_argument("--study-id", help="Limit to a single study ID")
    args = parser.parse_args()

    db = firestore.Client()

    if args.study_id:
        studies = [{"id": args.study_id, "name": args.study_id}]
    else:
        print("Fetching studies...")
        studies = fetch_all_studies(args.token)
        print(f"  {len(studies)} studies found\n")

    for study in studies:
        study_id = study["id"]
        study_title = study.get("name", study_id)

        submissions = fetch_submissions(study_id, args.token)
        approved_pids = {
            s["participant_id"]
            for s in submissions
            if s.get("status") == "APPROVED" and s.get("participant_id")
        }

        if not approved_pids:
            print(f"Study: {study_title}\n  No approved submissions.\n")
            continue

        # Load participant docs to get dataset/task
        participants = (
            db.collection("participants")
            .where(filter=FieldFilter("study_id", "==", study_id))
            .stream()
        )
        participant_docs = [p.to_dict() for p in participants if p.to_dict().get("prolific_pid") in approved_pids]

        print(f"Study: {study_title} ({study_id})")
        print(f"  Approved: {len(approved_pids)}  |  In DB: {len(participant_docs)}")
        print(f"\n  {'PID':<26}  {'Dataset':<12}  {'Task':<10}  {'Accuracy':>10}  {'Dur(min)':>8}  {'Video':>5}  {'Opens':>5}")
        print("  " + "-" * 80)

        total_correct = 0
        total_anns = 0

        for p in sorted(participant_docs, key=lambda d: (d.get("dataset",""), d.get("task",""))):
            pid = p["prolific_pid"]
            dataset = p.get("dataset", "?")
            task = p.get("task", "?")
            scene_id = p.get("scene_id", "main")
            created = p.get("created_at", "")[:19]
            completed = p.get("completed_at", "")
            video_played = p.get("video_played", False)

            # Duration from created to completed
            dur_str = "?"
            if created and completed:
                try:
                    c = datetime.fromisoformat(p["created_at"])
                    d = datetime.fromisoformat(p["completed_at"])
                    dur_str = f"{(d - c).seconds / 60:.1f}"
                except Exception:
                    pass

            anns = list(
                db.collection("annotations")
                .where(filter=FieldFilter("prolific_pid", "==", pid))
                .where(filter=FieldFilter("dataset", "==", dataset))
                .where(filter=FieldFilter("task", "==", task))
                .where(filter=FieldFilter("scene_id", "==", scene_id))
                .stream()
            )
            ann_dicts = [a.to_dict() for a in anns]
            correct, total = compute_accuracy(ann_dicts)
            total_correct += correct
            total_anns += total

            total_opens = sum(a.get("viewer_opens", 0) for a in ann_dicts if a.get("viewer_opens"))
            acc_str = f"{correct}/{total} ({100*correct/total:.0f}%)" if total else "0/0"

            print(f"  {pid:<26}  {dataset:<12}  {task:<10}  {acc_str:>10}  {dur_str:>8}  {'Y' if video_played else 'N':>5}  {total_opens:>5}")

        if total_anns:
            print(f"\n  Overall accuracy: {total_correct}/{total_anns} ({100*total_correct/total_anns:.1f}%)")
        print()


if __name__ == "__main__":
    main()
