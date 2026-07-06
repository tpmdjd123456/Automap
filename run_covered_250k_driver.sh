#!/bin/bash
# Self-contained driver for the covered-250k experiment. Runs server-side under
# setsid so it survives SSH disconnects. Builds the sample, verifies every
# benchmark-reachable pair is present (30134), then runs the pipeline.
cd /home/automap/Automap || exit 1
PY=.venv/bin/python

echo "[driver $(date)] === building covered-250k sample (all cover tables + filler) ==="
$PY -u build_covered_sample_stream.py \
    data/wdc_benchmark_cover.jsonl 250000 data/wdc_covered_250k.jsonl \
    data/wdc_archives/00.tar.gz data/wdc_archives/01.tar.gz

echo "[driver $(date)] === verifying coverage (expect 30134) ==="
COV=$($PY -u extract_covered_pairs.py data/benchmark-web.txt \
        data/wdc_covered_250k.jsonl /tmp/cov_250k.txt 2>&1 \
        | grep "Covered gold pairs" | grep -oE "[0-9]+$")
echo "[driver $(date)] COVERAGE=$COV"

if [ "$COV" != "30134" ]; then
    echo "[driver $(date)] COVERAGE MISMATCH ($COV != 30134) — ABORTING, not launching pipeline"
    exit 1
fi

echo "[driver $(date)] coverage OK — launching pipeline (jaccard, cap=250)"
mkdir -p output/wdc_covered_250k
$PY -u main.py \
    --corpus_path data/wdc_covered_250k.jsonl \
    --output_folder output/wdc_covered_250k/ \
    --threshold 0.3 --theta 0.95 \
    --parallel_workers 8 \
    --max_bucket_size 250 \
    --no_save_index \
    --string_matcher jaccard \
    > output/wdc_covered_250k.log 2>&1
echo "[driver $(date)] pipeline exited code $? — DONE"
