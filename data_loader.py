"""Loader for table corpora.

Supports two input formats, auto-dispatched on path:
- JSON Lines file (`.json` / `.jsonl`): one web-table object per line.
- Folder of CSVs: each `.csv` becomes one table.

The loader produces a list of `Table = (metadata: dict, columns: list[list[str]])`
tuples where columns are already normalized (strip + lowercase + whitespace
collapse) and short columns (<2 unique values, empty) are dropped.
"""

from __future__ import annotations

import csv
import json
import math
import os
import re
from typing import Any, Dict, Iterable, List, Tuple

_WS_RE = re.compile(r"\s+")

Table = Tuple[Dict[str, Any], List[List[str]]]


def clean_value(value: Any) -> str:
    """Normalize a single cell value.

    - None / NaN  -> ""
    - Any value cast to str, then stripped, lowercased, and internal
      whitespace runs collapsed to a single space.

    Returns a string (never None).
    """
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    s = str(value).strip().lower()
    return _WS_RE.sub(" ", s)


def _load_jsonl(
    path: str,
    table_types: Iterable[str] = ("RELATION",),
    strip_headers: bool = True,
) -> List[Table]:
    """Load tables from a JSON Lines file.

    Each line is a JSON object with at minimum:
      - `relation`: column-major list[list[Any]]
      - `tableType`: string
      - `hasHeader`: bool
      - `headerRowIndex`: int  (-1 when no header)

    Tables whose `tableType` is not in `table_types` are skipped. When
    `strip_headers` is True and `hasHeader` is True, the row at
    `headerRowIndex` is removed from every column. Values pass through
    `clean_value`. Columns that are empty or have <2 unique values are
    dropped. The returned metadata dict is the input record minus the
    `relation` field (which is replaced by the cleaned columns).
    """
    keep = set(table_types)
    out: List[Table] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("tableType") not in keep:
                continue
            relation = rec.get("relation") or []
            if strip_headers and rec.get("hasHeader") and rec.get("headerRowIndex", -1) >= 0:
                hri = rec["headerRowIndex"]
                relation = [
                    [v for i, v in enumerate(col) if i != hri]
                    for col in relation
                ]
            cleaned: List[List[str]] = []
            for col in relation:
                vals = [clean_value(v) for v in col]
                if len({v for v in vals if v != ""}) < 2:
                    continue
                cleaned.append(vals)
            if not cleaned:
                continue
            metadata = {k: v for k, v in rec.items() if k != "relation"}
            out.append((metadata, cleaned))
    return out


def _load_csv_folder(path: str) -> List[Table]:
    """Load every `.csv` file in `path` as a separate table.

    CSVs are row-major; we transpose to column-major. Values pass through
    `clean_value`. Columns that are empty or have <2 unique values are
    dropped. CSV files carry no metadata, so the returned metadata dict
    contains only `{"source": <filename>}` for traceability.
    """
    out: List[Table] = []
    for name in sorted(os.listdir(path)):
        if not name.lower().endswith(".csv"):
            continue
        fp = os.path.join(path, name)
        with open(fp, "r", encoding="utf-8", errors="ignore", newline="") as f:
            rows = list(csv.reader(f))
        if not rows:
            continue
        width = max(len(r) for r in rows)
        columns_raw: List[List[str]] = [[] for _ in range(width)]
        for row in rows:
            for i in range(width):
                columns_raw[i].append(row[i] if i < len(row) else "")
        cleaned: List[List[str]] = []
        for col in columns_raw:
            vals = [clean_value(v) for v in col]
            if len({v for v in vals if v != ""}) < 2:
                continue
            cleaned.append(vals)
        if cleaned:
            out.append(({"source": name}, cleaned))
    return out


def load_corpus(
    path: str,
    *,
    table_types: Iterable[str] = ("RELATION",),
    strip_headers: bool = True,
) -> List[Table]:
    """Auto-dispatch on path: file -> JSONL backend, dir -> CSV-folder backend.

    Args:
        path: filesystem path. `*.json`/`*.jsonl` files use the JSONL
            backend; directories use the CSV-folder backend.
        table_types: JSONL only - which `tableType` values to keep.
        strip_headers: JSONL only - whether to drop the header row.

    Returns: list of (metadata, columns) tuples.
    """
    if os.path.isdir(path):
        return _load_csv_folder(path)
    if os.path.isfile(path):
        return _load_jsonl(path, table_types=table_types, strip_headers=strip_headers)
    raise FileNotFoundError(f"Corpus path does not exist: {path}")


def corpus_summary(corpus: List[Table]) -> None:
    """Print summary statistics for a loaded corpus to stdout."""
    n_tables = len(corpus)
    n_cols = sum(len(cols) for _, cols in corpus)
    unique_values: set = set()
    for _, cols in corpus:
        for col in cols:
            unique_values.update(col)
    avg_cols = n_cols / n_tables if n_tables else 0
    print(f"  Loaded {n_tables} tables, {n_cols} columns, {len(unique_values)} unique values")
    print(f"  Avg columns per table: {avg_cols:.2f}")
