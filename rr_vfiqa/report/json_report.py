"""JSON report writer."""

from __future__ import annotations

import json
from pathlib import Path

from ..schema import Report


def write_json_report(report: Report, path: str | Path, indent: int = 2) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report.to_dict(), indent=indent, ensure_ascii=False),
                    encoding="utf-8")
    return path
