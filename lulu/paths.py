"""Project locations; importing these helpers never loads ML dependencies."""
from __future__ import annotations

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_OUTPUT_ROOT = Path(
    os.environ.get("LULU_OUTPUT_ROOT", str(PROJECT_ROOT / "outputs"))
).expanduser().resolve()

DEFAULT_BENCHMARK_PARSER = PROJECT_ROOT / "lulu" / "benchmark_parser.py"
PROJECT_SCRIPTS = PROJECT_ROOT / "scripts"


def benchmark_parser_path(value: str | Path | None = None) -> Path:
    """Resolve Lulu's bundled benchmark parser."""
    location = value if value is not None else os.environ.get("LULU_BENCHMARK_PARSER")
    path = (
        Path(location).expanduser().resolve()
        if location
        else DEFAULT_BENCHMARK_PARSER
    )
    if not path.is_file():
        raise FileNotFoundError(path)
    return path