"""Server-side filter + 30k-table sample + extract to JSONL.

Runs ON dama (big-dama-3), connects directly to Vertica on big-dama-1.
Streams the result rows, pivots each tableid into a column-major `relation`,
and writes one JSONL line per surviving sampled table.

Output schema matches what data_loader._load_jsonl expects:
    {relation: [[col_vals,...],...],
     tableType: "RELATION", hasHeader: False, headerRowIndex: -1,
     tableid: int}
"""

from __future__ import annotations
import argparse, json, sys, time
import vertica_python

CONN = {
    "host": "big-dama-1.dima.tu-berlin.de",
    "port": 5433,
    "user": "automap",
    "password": "automapd2ip",
    "database": "xformer",
    "autocommit": True,
    "read_timeout": 36000,  # 10 hours
}

# One query: filter columns + filter tables + sample 30k + project rows
SQL = """
WITH stripped AS (
    SELECT tableid, colid, rowid, TRIM(BOTH '''' FROM term) AS val
    FROM public.main_tokenized
),
col_stats AS (
    SELECT
        tableid, colid,
        COUNT(DISTINCT CASE WHEN val <> '' THEN val END) AS uniq,
        SUM(CASE WHEN val <> '' AND NOT REGEXP_LIKE(val, '^-?[0-9]+([.,][0-9]+)?$') THEN 1 ELSE 0 END) AS non_num,
        SUM(CASE WHEN val <> '' AND NOT REGEXP_LIKE(val, '^(#|0x)[0-9a-fA-F]+$') THEN 1 ELSE 0 END) AS non_hex,
        SUM(CASE WHEN val <> '' AND LOWER(val) NOT IN ('-','--','n/a','na','null','none','?') THEN 1 ELSE 0 END) AS non_ph
    FROM stripped
    GROUP BY tableid, colid
),
keep_cols AS (
    SELECT tableid, colid
    FROM col_stats
    WHERE uniq >= 2 AND non_num > 0 AND non_hex > 0 AND non_ph > 0
),
surviving_tables AS (
    SELECT tableid FROM keep_cols GROUP BY tableid HAVING COUNT(*) >= 2
),
sampled AS (
    SELECT tableid FROM surviving_tables ORDER BY HASH(tableid) LIMIT {n}
)
SELECT mt.tableid, mt.colid, mt.rowid, TRIM(BOTH '''' FROM mt.term) AS val
FROM public.main_tokenized mt
INNER JOIN keep_cols kc ON mt.tableid = kc.tableid AND mt.colid = kc.colid
INNER JOIN sampled s ON mt.tableid = s.tableid
ORDER BY mt.tableid, mt.colid, mt.rowid
"""


def flush_table(out, tid, by_col):
    """Write one JSONL line for a completed tableid."""
    if not by_col:
        return False
    cols_sorted = sorted(by_col.keys())
    n_rows = max((max(rows.keys()) for rows in by_col.values()), default=-1) + 1
    if n_rows < 1:
        return False
    relation = []
    for colid in cols_sorted:
        rows = by_col[colid]
        col_vals = [rows.get(r, "") for r in range(n_rows)]
        relation.append(col_vals)
    rec = {
        "relation": relation,
        "tableType": "RELATION",
        "hasHeader": False,
        "headerRowIndex": -1,
        "tableid": tid,
    }
    out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    print(f"[{time.strftime('%H:%M:%S')}] Connecting to Vertica...", flush=True)
    t0 = time.time()
    with vertica_python.connect(**CONN) as conn:
        cur = conn.cursor()
        print(f"[{time.strftime('%H:%M:%S')}] Submitting filter+sample+extract query (n={args.n})...", flush=True)
        t_q = time.time()
        cur.execute(SQL.format(n=args.n))
        print(f"[{time.strftime('%H:%M:%S')}] Query accepted, streaming results...", flush=True)

        rows_seen = 0
        tables_written = 0
        cur_tid = None
        by_col: dict[int, dict[int, str]] = {}

        with open(args.out, "w", encoding="utf-8") as out:
            for tid, cid, rid, val in cur.iterate():
                rows_seen += 1
                if tid != cur_tid:
                    if cur_tid is not None:
                        if flush_table(out, cur_tid, by_col):
                            tables_written += 1
                    cur_tid = tid
                    by_col = {}
                by_col.setdefault(cid, {})[rid] = val or ""
                if rows_seen % 100000 == 0:
                    print(f"  {rows_seen:,} rows | {tables_written:,} tables written", flush=True)
            if cur_tid is not None:
                if flush_table(out, cur_tid, by_col):
                    tables_written += 1
        cur.close()

    t_total = time.time() - t0
    print(f"\n[{time.strftime('%H:%M:%S')}] DONE: streamed {rows_seen:,} rows, wrote {tables_written:,} tables", flush=True)
    print(f"  total time: {t_total:.1f}s ({t_total/60:.1f} min)", flush=True)
    print(f"  query+stream time: {time.time()-t_q:.1f}s", flush=True)
    print(f"  output: {args.out}", flush=True)


if __name__ == "__main__":
    main()
