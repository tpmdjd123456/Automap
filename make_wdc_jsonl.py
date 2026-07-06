#!/usr/bin/env python3
"""Stream the first N web-table records out of a (nested) WDC tar.gz into a
single JSON Lines file, without extracting the ~1M tiny .json files to disk.

The WDC archives are nested: outer `NN.tar.gz` contains one `NN.tar`, which
contains directories of one-table-per-file `*.json` records. Each record is
already in the schema data_loader.py expects (relation/tableType/hasHeader/...).

Usage:
    python make_wdc_jsonl.py <archive.tar.gz> <N> <out.jsonl>

N <= 0 means "all records".
"""

from __future__ import annotations

import json
import sys
import tarfile


def main() -> None:
    archive, n_str, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
    limit = int(n_str)
    written = 0
    seen = 0

    outer = tarfile.open(archive, "r:gz")
    try:
        for outer_m in outer:
            if not outer_m.isfile():
                continue
            inner_fo = outer.extractfile(outer_m)
            if inner_fo is None:
                continue
            # Stream the inner tar sequentially.
            inner = tarfile.open(fileobj=inner_fo, mode="r|")
            with open(out_path, "w", encoding="utf-8") as out:
                for tm in inner:
                    if not tm.name.endswith(".json"):
                        continue
                    seen += 1
                    f = inner.extractfile(tm)
                    if f is None:
                        continue
                    try:
                        rec = json.loads(f.read())
                    except Exception:
                        continue
                    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    written += 1
                    if written % 50000 == 0:
                        print(f"  ... {written} records written", flush=True)
                    if limit > 0 and written >= limit:
                        break
            inner.close()
            break  # only the single inner tar
    finally:
        outer.close()

    print(f"Done: scanned {seen} files, wrote {written} records to {out_path}",
          flush=True)


if __name__ == "__main__":
    main()
