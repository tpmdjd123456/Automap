#!/usr/bin/env bash
# WDC web-table calibration using the EXACT 1.5M Vertica run config
# (max_bucket_size cap + jaccard matcher) so we can project the full ~2M run.
set -euo pipefail

cd /home/automap/Automap
PY=.venv/bin/python
ARCHIVE=data/wdc_archives/00.tar.gz
WORKERS=8            # match 1.5M run
MAXBUCKET=250        # the heuristic that bounds the O(k^2) blowup
STAMP=$(date +%Y%m%d_%H%M%S)
LOGDIR=output/wdc_calib_capped_${STAMP}
mkdir -p "$LOGDIR"

# Reuse the 250k JSONL if already built; otherwise stream it from the archive.
if [ ! -s data/wdc_250k.jsonl ]; then
  echo "[$(date)] Building 250k-record JSONL from $ARCHIVE ..."
  python3 -u make_wdc_jsonl.py "$ARCHIVE" 250000 data/wdc_250k.jsonl 2>&1 | tee "$LOGDIR/convert.log"
fi
head -n 10000 data/wdc_250k.jsonl > data/wdc_10k.jsonl
head -n 50000 data/wdc_250k.jsonl > data/wdc_50k.jsonl
wc -l data/wdc_10k.jsonl data/wdc_50k.jsonl data/wdc_250k.jsonl

for SIZE in 10k 50k 250k; do
  CORPUS=data/wdc_${SIZE}.jsonl
  OUT=output/wdc_${SIZE}_capped
  LOG=$LOGDIR/run_${SIZE}.log
  mkdir -p "$OUT"
  echo "[$(date)] ===== RUN ${SIZE} (cap=$MAXBUCKET, workers=$WORKERS) ====="
  /usr/bin/time -v "$PY" -u main.py \
    --corpus_path "$CORPUS" \
    --output_folder "$OUT/" \
    --threshold 0.3 --theta 0.95 \
    --parallel_workers "$WORKERS" \
    --max_bucket_size "$MAXBUCKET" \
    --string_matcher jaccard \
    --no_save_index \
    2>&1 | tee "$LOG"
  echo "[$(date)] ===== DONE ${SIZE} ====="
done

echo "[$(date)] Calibration complete."
