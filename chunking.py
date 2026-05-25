import json
import os
from collections import Counter
from typing import List, Dict, Any, Tuple

Table = Tuple[Dict[str, Any], List[List[str]]]

# ---------------------------------------------------------------------------
# Numeric / Null detection
# ---------------------------------------------------------------------------

NULL_TOKENS = {"", "n/a", "na", "null", "none", "-", "—", "–", "?", "unknown"}

NUMERIC_THRESHOLD = 0.5   # drop column if ≥ this fraction of non-null values contain a digit
NULL_THRESHOLD    = 0.5   # drop column if ≥ this fraction of all values are null/empty


def _is_numeric(value: str) -> bool:
    """Any value containing a digit is treated as numeric."""
    return any(ch.isdigit() for ch in value)


def _is_null(value: str) -> bool:
    return value.strip().lower() in NULL_TOKENS


def _is_string_column(vals: List[str]) -> bool:
    """Return True only if the column is predominantly non-null, non-numeric strings."""
    if not vals:
        return False

    null_frac = sum(1 for v in vals if _is_null(v)) / len(vals)
    if null_frac >= NULL_THRESHOLD:
        return False

    non_null = [v for v in vals if not _is_null(v)]
    if not non_null:
        return False

    numeric_frac = sum(1 for v in non_null if _is_numeric(v)) / len(non_null)
    return numeric_frac < NUMERIC_THRESHOLD


# ---------------------------------------------------------------------------
# Cleaning & loading
# ---------------------------------------------------------------------------

def clean_value(v: str) -> str:
    """Cleans up individual text strings and filters out uninformative filler."""
    if not v:
        return ""
    t = v.strip().lower()
    if t in ["", "?", "—", "-", "total", "null", "none", "unknown", "x"]:
        return ""
    return v.strip()


def _load_and_transform_line(line: str) -> Table | None:
    """Processes a single raw line, handles structural conversion, and filters noise."""
    try:
        raw_rec = json.loads(line.strip())
    except json.JSONDecodeError:
        return None

    # PHASE 1: LAYOUT TRANSFORMATION (Extract Text & Pivot to Column-Major)
    raw_rows = raw_rec.get("tableData", [])
    if not raw_rows:
        return None

    text_rows = [[cell.get("text", "") for cell in row] for row in raw_rows]
    column_major = [list(col) for col in zip(*text_rows)] if text_rows else []

    # PHASE 2: DATA CLEANING & STATISTICAL FILTERING
    cleaned_columns: List[List[str]] = []

    for col in column_major:
        vals = [clean_value(v) for v in col]
        non_empty_vals = [v for v in vals if v != ""]

        # Drop column if it's mostly empty
        if len(non_empty_vals) < 3:
            continue

        # Variance Filter: Drop constant columns (e.g., "Ulster Unionist")
        val_counts = Counter(non_empty_vals)
        max_count = val_counts.most_common(1)[0][1]
        if (max_count / len(non_empty_vals)) > 0.70:
            continue

        # ID Filter: Drop purely sequential integers (1, 2, 3...)
        try:
            int_vals = [int(v) for v in non_empty_vals]
            if int_vals == list(range(int_vals[0], int_vals[0] + len(int_vals))):
                continue
        except ValueError:
            pass

        # Uniqueness Filter: Column must contain at least 2 unique values
        if len(set(non_empty_vals)) < 2:
            continue

        # Numeric / Null Filter: Drop columns where most values contain digits
        if not _is_string_column(vals):
            continue

        cleaned_columns.append(vals)

    if not cleaned_columns:
        return None

    # PHASE 3: METADATA WRAPPING
    metadata = {
        "_id":          raw_rec.get("_id"),
        "pgId":         raw_rec.get("pgId"),
        "pgTitle":      raw_rec.get("pgTitle"),
        "sectionTitle": raw_rec.get("sectionTitle"),
        "tableCaption": raw_rec.get("tableCaption"),
        "tableType":    "RELATION",
        "hasHeader":    True if raw_rec.get("tableHeaders") else False,
        "headerRowIndex": -1,
    }

    return metadata, cleaned_columns


# =====================================================================
# EXECUTION PART
# =====================================================================
if __name__ == "__main__":
    input_file = "tables.json"
    output_dir = "dev_chunks"
    output_file = os.path.join(output_dir, "chunk_1.json")

    os.makedirs(output_dir, exist_ok=True)

    chunk_size = 5000
    valid_tables = []

    print("Starting processing... filtering noise and restructuring table schemas...")

    with open(input_file, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue

            result = _load_and_transform_line(line)

            if result is not None:
                metadata, cleaned_columns = result
                saved_format = {**metadata, "relation": cleaned_columns}
                valid_tables.append(saved_format)

            if len(valid_tables) >= chunk_size:
                break

    print(f"Writing parsed tables to {output_file}...")
    with open(output_file, "w", encoding="utf-8") as out_f:
        for table in valid_tables:
            out_f.write(json.dumps(table) + "\n")

    print(f"\n[SUCCESS] Extracted and saved exactly {len(valid_tables)} cleaned tables!")