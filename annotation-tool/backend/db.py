"""
SQLite access layer.

WAL mode + busy_timeout make concurrent participants safe without
requiring a connection pool. Writes are idempotent via ON CONFLICT upserts.
"""
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator, Union


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.execute("PRAGMA foreign_keys=ON")
    return conn


@contextmanager
def get_conn(db_path: Path) -> Generator[sqlite3.Connection, None, None]:
    conn = _connect(db_path)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db(db_path: Path) -> None:
    """Create tables if they don't exist."""
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with get_conn(db_path) as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS participants (
                prolific_pid  TEXT NOT NULL,
                dataset       TEXT NOT NULL,
                task          TEXT NOT NULL,
                seed          INTEGER NOT NULL,
                created_at    TEXT NOT NULL DEFAULT (datetime('now')),
                completed_at  TEXT,
                PRIMARY KEY (prolific_pid, dataset, task)
            );

            CREATE TABLE IF NOT EXISTS annotations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                prolific_pid  TEXT NOT NULL,
                dataset       TEXT NOT NULL,
                task          TEXT NOT NULL,
                sample_index  INTEGER NOT NULL,
                sample_name   TEXT NOT NULL,
                n_frames      INTEGER NOT NULL,
                ground_truth  TEXT NOT NULL,
                human_answer  TEXT NOT NULL,
                answered_at   TEXT NOT NULL DEFAULT (datetime('now')),
                UNIQUE(prolific_pid, dataset, task, sample_index)
            );
        """)


def upsert_participant(db_path: Path, prolific_pid: str, dataset: str, task: str, seed: int) -> None:
    """Insert or ignore a participant row (first visit wins)."""
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO participants (prolific_pid, dataset, task, seed)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(prolific_pid, dataset, task) DO NOTHING
            """,
            (prolific_pid, dataset, task, seed),
        )


def upsert_annotation(
    db_path: Path,
    prolific_pid: str,
    dataset: str,
    task: str,
    sample_index: int,
    sample_name: str,
    n_frames: int,
    ground_truth: Union[bool, tuple[int, int]],
    human_answer: Union[bool, tuple[int, int]],
) -> None:
    """Insert or update an annotation row (idempotent)."""
    gt_str = _encode(ground_truth)
    ha_str = _encode(human_answer)
    now = datetime.now(timezone.utc).isoformat()
    with get_conn(db_path) as conn:
        conn.execute(
            """
            INSERT INTO annotations
                (prolific_pid, dataset, task, sample_index, sample_name,
                 n_frames, ground_truth, human_answer, answered_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(prolific_pid, dataset, task, sample_index)
            DO UPDATE SET human_answer=excluded.human_answer, answered_at=excluded.answered_at
            """,
            (prolific_pid, dataset, task, sample_index, sample_name,
             n_frames, gt_str, ha_str, now),
        )


def mark_completed(db_path: Path, prolific_pid: str, dataset: str, task: str) -> None:
    now = datetime.now(timezone.utc).isoformat()
    with get_conn(db_path) as conn:
        conn.execute(
            """
            UPDATE participants SET completed_at=?
            WHERE prolific_pid=? AND dataset=? AND task=?
            """,
            (now, prolific_pid, dataset, task),
        )


def count_annotations(db_path: Path, prolific_pid: str, dataset: str, task: str) -> int:
    with get_conn(db_path) as conn:
        row = conn.execute(
            "SELECT COUNT(*) FROM annotations WHERE prolific_pid=? AND dataset=? AND task=?",
            (prolific_pid, dataset, task),
        ).fetchone()
    return row[0] if row else 0


# ── Encoding helpers ──────────────────────────────────────────────────────────

def _encode(value: Union[bool, tuple[int, int]]) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    return f"{value[0]},{value[1]}"


def decode_answer(raw: str, task: str) -> Union[bool, tuple[int, int]]:
    """Parse a stored answer string back to the appropriate Python type."""
    if task == "detect":
        return raw.lower() == "true"
    parts = raw.split(",")
    return (int(parts[0]), int(parts[1]))
