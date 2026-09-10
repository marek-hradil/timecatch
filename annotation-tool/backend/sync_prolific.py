#!/usr/bin/env python3
"""
Sync Prolific submission statuses to Firestore participant docs.

Fetches all studies and their submissions from the Prolific API, then writes
`prolific_status` (e.g. APPROVED, REJECTED, RETURNED) to each matching
participant doc. Never deletes or overwrites any other fields.

Usage:
    python3 sync_prolific.py --token <prolific_api_token> [--dry-run]
    python3 sync_prolific.py --token <prolific_api_token> --study-id <id> [--dry-run]

Get your API token at: https://app.prolific.com/researcher/settings/api
"""
import argparse
import requests
from google.cloud import firestore

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
    # Try project endpoint first; fall back to global list filtered by project
    try:
        data = prolific_get(f"/projects/{PROJECT_ID}/studies/", token)
        results = data.get("results", [])
        if results or "results" in data:
            return results
    except requests.HTTPError:
        pass
    # Fallback: fetch all studies and filter by project_id
    data = prolific_get("/studies/", token)
    return [s for s in data.get("results", []) if s.get("project") == PROJECT_ID]


def fetch_submissions(study_id: str, token: str) -> list[dict]:
    results = []
    path = f"/studies/{study_id}/submissions/?limit=100"
    while path:
        data = prolific_get(path, token)
        results.extend(data.get("results", []))
        next_url = data.get("next")
        if next_url:
            path = next_url.replace(PROLIFIC_API, "")
        else:
            path = None
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--token", required=True, help="Prolific API token")
    parser.add_argument("--study-id", help="Sync a single study ID instead of the whole project")
    parser.add_argument("--dry-run", action="store_true", help="Print changes without writing to Firestore")
    args = parser.parse_args()

    db = firestore.Client()
    participants_col = db.collection("participants")

    print("Loading participant docs from Firestore...")
    all_participants = list(participants_col.stream())
    pid_to_docs: dict[str, list] = {}
    for doc in all_participants:
        d = doc.to_dict()
        pid = d.get("prolific_pid")
        if pid:
            pid_to_docs.setdefault(pid, []).append(doc.reference)
    print(f"  {len(all_participants)} participant docs, {len(pid_to_docs)} unique PIDs")

    if args.study_id:
        studies = [{"id": args.study_id, "name": args.study_id}]
        print(f"Targeting single study: {args.study_id}")
    else:
        print("Fetching studies from Prolific...")
        studies = fetch_all_studies(args.token)
        print(f"  {len(studies)} studies found:")
        for s in studies:
            print(f"    {s.get('name', '?')} ({s['id']})")

    updated = 0
    skipped = 0

    for study in studies:
        study_id = study["id"]
        study_title = study.get("name", study_id)
        print(f"\nStudy: {study_title} ({study_id})")

        try:
            submissions = fetch_submissions(study_id, args.token)
        except requests.HTTPError as e:
            print(f"  Skipping — API error: {e}")
            continue

        print(f"  {len(submissions)} submissions")
        for sub in submissions:
            pid = sub.get("participant_id")
            status = sub.get("status")
            if not pid or not status:
                continue

            docs = pid_to_docs.get(pid, [])
            if not docs:
                skipped += 1
                continue

            for doc_ref in docs:
                if args.dry_run:
                    print(f"  [dry-run] {pid} → {status}  ({doc_ref.id})")
                else:
                    doc_ref.set({"prolific_status": status}, merge=True)
                updated += 1

    print(f"\nDone. Updated: {updated}  Skipped (not in DB): {skipped}")
    if args.dry_run:
        print("(dry-run — no changes written)")


if __name__ == "__main__":
    main()
