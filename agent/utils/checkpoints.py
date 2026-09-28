"""Checkpoint discovery.

The deploy scripts previously searched ``./models/steps``, a directory no script
ever creates, so the numbered-checkpoint fallback could never find anything.
Resolution is now rooted at the profile's model directory and compares indices
numerically.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import List, Optional


def get_last_index(directory: Path, prefix: str, suffix: str = ".zip") -> int:
    """Highest numeric index in ``<prefix><N><suffix>``, or -1 if none match."""
    directory = Path(directory)
    if not directory.exists():
        return -1
    pattern = re.compile(re.escape(prefix) + r"(\d+)" + re.escape(suffix) + r"$")
    indices = [int(m.group(1)) for f in directory.iterdir() if f.is_file()
               for m in [pattern.search(f.name)] if m]
    return max(indices, default=-1)


def list_checkpoints(directory: Path, prefix: str = "bc_policy",
                     suffix: str = ".zip") -> List[Path]:
    directory = Path(directory)
    if not directory.exists():
        return []
    pattern = re.compile(re.escape(prefix) + r"(\d*)" + re.escape(suffix) + r"$")
    matches = []
    for f in directory.iterdir():
        m = pattern.search(f.name)
        if f.is_file() and m:
            matches.append((int(m.group(1)) if m.group(1) else -1, f))
    return [f for _, f in sorted(matches, key=lambda pair: pair[0], reverse=True)]


def resolve_checkpoint(directory: Path, prefix: str = "bc_policy",
                       suffix: str = ".zip") -> Optional[Path]:
    """Prefer the final ``<prefix>.zip``, else the highest numbered checkpoint."""
    directory = Path(directory)
    final = directory / f"{prefix}{suffix}"
    if final.exists():
        return final
    candidates = list_checkpoints(directory, prefix, suffix)
    return candidates[0] if candidates else None
