"""JSON / JSONL file helpers used across modules."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_json(filepath: str | Path) -> Any:
    """Load and return parsed JSON from *filepath*."""
    with open(filepath, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_json(data: Any, filepath: str | Path, *, indent: int = 2) -> Path:
    """Serialise *data* as JSON to *filepath* and return the path."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=indent, ensure_ascii=False)
    return path


def load_jsonl(filepath: str | Path) -> list[dict[str, Any]]:
    """Load a JSON-Lines file and return a list of dicts."""
    records: list[dict[str, Any]] = []
    with open(filepath, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def save_jsonl(
    data: list[dict[str, Any]],
    filepath: str | Path,
) -> Path:
    """Write a list of dicts as JSON-Lines to *filepath*."""
    path = Path(filepath)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for record in data:
            fh.write(json.dumps(record, ensure_ascii=False) + "\n")
    return path
